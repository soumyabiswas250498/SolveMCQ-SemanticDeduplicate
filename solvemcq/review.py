import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from .input import stable_json


Verdict = Literal["duplicate", "distinct", "needs_review"]


class CandidateQuestion(BaseModel):
    input_row: int | None
    subject: str
    question: str
    answer: str
    mongo_id: str | None = None
    mongo_id_type: Literal["ObjectId", "str"] | None = None


class CandidatePair(BaseModel):
    question_ids: tuple[str, str]
    question_score: float = Field(ge=-1, le=1)
    answer_score: float = Field(ge=-1, le=1)
    same_answer_text: bool


class CandidateReport(BaseModel):
    schema_version: Literal[1, 2]
    batch_id: str
    questions: dict[str, CandidateQuestion]
    pairs: list[CandidatePair]
    existing_ids: dict[int, str] | None = None


class Decision(BaseModel):
    pair_id: int = Field(ge=1)
    verdict: Verdict
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    keep_question_id: str | None = None


class ReviewJob:
    def __init__(self, report_path: Path, db_path: Path, *, read_only: bool = False):
        self.report = CandidateReport.model_validate_json(report_path.read_bytes())
        for question in self.report.questions.values():
            has_mongo_id = question.mongo_id is not None
            if (
                has_mongo_id != (question.input_row is None)
                or has_mongo_id != (question.mongo_id_type is not None)
                or (has_mongo_id and self.report.schema_version != 2)
            ):
                raise ValueError("Candidate question has invalid source metadata")
        seen = set()
        for pair in self.report.pairs:
            left, right = pair.question_ids
            questions = self.report.questions
            key = frozenset(pair.question_ids)
            if (
                left == right or left not in questions or right not in questions
                or key in seen or questions[left].subject != questions[right].subject
            ):
                raise ValueError(f"Invalid or repeated candidate pair: {pair.question_ids}")
            seen.add(key)

        # A changed candidate set must never inherit decisions for positional pair IDs.
        report_data = self.report.model_dump(
            mode="json", exclude_none=self.report.schema_version == 1,
        )
        if self.report.existing_ids is None:
            report_data.pop("existing_ids", None)
        report_json = stable_json(report_data)
        self.job_id = hashlib.sha256(report_json.encode("utf-8")).hexdigest()
        self.db_path = db_path
        if read_only:
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS review_decisions (
                    job_id TEXT NOT NULL,
                    pair_id INTEGER NOT NULL,
                    verdict TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    keep_question_id TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (job_id, pair_id)
                )
            """)

    def saved_decisions(self):
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            return {
                pair_id: Decision(
                    pair_id=pair_id, verdict=verdict, reason=reason, keep_question_id=keeper,
                )
                for pair_id, verdict, reason, keeper in db.execute("""
                    SELECT pair_id, verdict, reason, keep_question_id
                    FROM review_decisions WHERE job_id = ? ORDER BY pair_id
                """, (self.job_id,))
            }

    def summary(self):
        with closing(sqlite3.connect(self.db_path)) as db:
            counts = dict(db.execute("""
                SELECT verdict, count(*) FROM review_decisions WHERE job_id = ? GROUP BY verdict
            """, (self.job_id,)))
        total = len(self.report.pairs)
        return {
            "job_id": self.job_id,
            "batch_id": self.report.batch_id,
            "candidate_pairs": total,
            "pending": total - sum(counts.values()),
            "duplicates": counts.get("duplicate", 0),
            "distinct": counts.get("distinct", 0),
            "needs_review": counts.get("needs_review", 0),
        }

    def batch(self, limit: int, queue: Literal["pending", "needs_review"], *, after: int = 0):
        with closing(sqlite3.connect(self.db_path)) as db:
            decisions = {pair_id: (verdict, reason) for pair_id, verdict, reason in db.execute("""
                SELECT pair_id, verdict, reason FROM review_decisions WHERE job_id = ?
            """, (self.job_id,))}
        candidates = []
        for pair_id in range(1, len(self.report.pairs) + 1):
            verdict, reason = decisions.get(pair_id, ("pending", None))
            if verdict != queue or pair_id <= after:
                continue
            candidates.append(self.describe_pair(pair_id, reason))
            if len(candidates) == limit:
                break
        return {"job_id": self.job_id, "queue": queue, "candidates": candidates}

    def describe_pair(self, pair_id, reason=None):
        pair = self.report.pairs[pair_id - 1]
        return {
            "pair_id": pair_id,
            **pair.model_dump(mode="json"),
            "questions": [
                {"question_id": question_id, **self.report.questions[question_id].model_dump(exclude_none=True)}
                for question_id in pair.question_ids
            ],
            "previous_reason": reason,
        }

    def submit(self, job_id: str, decisions: list[Decision], *, replace: Decision | None = None):
        if job_id != self.job_id:
            raise ValueError("Job ID does not match the current report. Fetch a new review batch.")
        seen = set()
        for decision in decisions:
            if decision.pair_id > len(self.report.pairs) or decision.pair_id in seen:
                raise ValueError(f"Unknown or repeated pair ID: {decision.pair_id}")
            seen.add(decision.pair_id)
            pair = self.report.pairs[decision.pair_id - 1]
            if decision.verdict == "duplicate":
                if decision.keep_question_id not in pair.question_ids:
                    raise ValueError(f"Pair {decision.pair_id} requires a keeper from that pair")
                existing = [
                    item for item in pair.question_ids
                    if self.report.questions[item].mongo_id is not None
                ]
                if len(existing) == 1 and decision.keep_question_id != existing[0]:
                    raise ValueError(f"Pair {decision.pair_id} must keep the existing MongoDB question")
            elif decision.keep_question_id is not None:
                raise ValueError("Only a duplicate verdict can name a keeper")

        saved = 0
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            for decision in decisions:
                key = (self.job_id, decision.pair_id)
                value = (decision.verdict, decision.reason, decision.keep_question_id)
                previous = db.execute("""
                    SELECT verdict, reason, keep_question_id
                    FROM review_decisions WHERE job_id = ? AND pair_id = ?
                """, key).fetchone()
                if previous == value:
                    continue
                if replace is not None:
                    expected = (replace.verdict, replace.reason, replace.keep_question_id)
                    if replace.pair_id != decision.pair_id or previous != expected:
                        raise ValueError(f"Pair {decision.pair_id} changed. Preview the results again.")
                elif previous is not None and previous[0] != "needs_review":
                    raise ValueError(f"Pair {decision.pair_id} already has a final decision")
                db.execute("""
                    INSERT INTO review_decisions (job_id, pair_id, verdict, reason, keep_question_id)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (job_id, pair_id) DO UPDATE SET
                        verdict = excluded.verdict,
                        reason = excluded.reason,
                        keep_question_id = excluded.keep_question_id,
                        updated_at = CURRENT_TIMESTAMP
                """, (*key, *value))
                saved += 1
        return {"job_id": self.job_id, "saved": saved, "already_saved": len(decisions) - saved}
