import hashlib
import json
from collections import Counter, defaultdict
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
            return self._decisions(db)

    def _decisions(self, db):
        return {
            pair_id: Decision(pair_id=pair_id, verdict=verdict, reason=reason, keep_question_id=keeper)
            for pair_id, verdict, reason, keeper in db.execute("""
                SELECT pair_id, verdict, reason, keep_question_id
                FROM review_decisions WHERE job_id = ? ORDER BY pair_id
            """, (self.job_id,))
        }

    def _selections(self, db):
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name = 'review_group_selections'").fetchone():
            return {}
        return {
            group_id: json.loads(data)
            for group_id, data in db.execute("""
                SELECT group_id, selection FROM review_group_selections WHERE job_id = ?
            """, (self.job_id,))
        }

    def review_state(self):
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            db.execute("BEGIN")
            return self._decisions(db), self._selections(db)

    def export_review_state(self):
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            db.execute("BEGIN")
            decisions, selections = self._decisions(db), self._selections(db)
            history = []
            if db.execute("SELECT 1 FROM sqlite_master WHERE name = 'review_attribution'").fetchone():
                history = db.execute("""
                    SELECT pair_id, source, verdict FROM review_attribution WHERE job_id = ?
                """, (self.job_id,)).fetchall()
        agent = Counter(verdict for _, source, verdict in history if source == "agent")
        recorded = {pair_id for pair_id, _, _ in history}
        legacy = Counter(decision.verdict for pair_id, decision in decisions.items() if pair_id not in recorded)
        metrics = {
            "agent_resolved_pairs": agent["duplicate"] + agent["distinct"],
            "agent_duplicate_pairs": agent["duplicate"],
            "agent_distinct_pairs": agent["distinct"],
            "agent_deferred_pairs": agent["needs_review"],
            "manual_groups_resolved": len(selections),
            "manual_pairs_resolved": sum(source == "manual" and verdict != "needs_review"
                                         for _, source, verdict in history),
            "unattributed_pairs": sum(legacy.values()),
            "unattributed_duplicate_pairs": legacy["duplicate"],
            "unattributed_distinct_pairs": legacy["distinct"],
            "unattributed_deferred_pairs": legacy["needs_review"],
        }
        return decisions, selections, metrics

    def summary(self):
        decisions, selections = self.review_state()
        covered = {pair_id for item in selections.values() for pair_id in item["pairs"]}
        counts = Counter(decision.verdict for pair_id, decision in decisions.items() if pair_id not in covered)
        total = len(self.report.pairs)
        result = {
            "job_id": self.job_id,
            "batch_id": self.report.batch_id,
            "candidate_pairs": total,
            "pending": total - sum(counts.values()) - len(covered),
            "duplicates": counts["duplicate"],
            "distinct": counts["distinct"],
            "needs_review": counts["needs_review"],
            "recorded_duplicate_verdicts": sum(item.verdict == "duplicate" for item in decisions.values()),
            "recorded_distinct_verdicts": sum(item.verdict == "distinct" for item in decisions.values()),
        }
        if covered:
            result["resolved_by_selection"] = len(covered)
        return result

    def batch(self, limit: int, queue: Literal["pending", "needs_review"], *, after: int = 0):
        decisions, selections = self.review_state()
        covered = {pair_id for item in selections.values() for pair_id in item["pairs"]}
        candidates = []
        for pair_id in range(after + 1, len(self.report.pairs) + 1):
            decision = decisions.get(pair_id)
            verdict = decision.verdict if decision else "pending"
            if verdict != queue or pair_id in covered:
                continue
            candidates.append(self.describe_pair(pair_id, decision.reason if decision else None))
            if len(candidates) == limit:
                break
        return {"job_id": self.job_id, "queue": queue, "candidates": candidates}

    def groups(self, decisions, selections):
        groups = {group_id: {**item, "id": group_id, "status": "saved"}
                  for group_id, item in selections.items()}
        saved_members = {question for item in selections.values() for question in item["members"]}
        edges = defaultdict(list)
        for pair_id, pair in enumerate(self.report.pairs, 1):
            left, right = pair.question_ids
            decision = decisions.get(pair_id)
            if left in saved_members or right in saved_members or (decision and decision.verdict == "distinct"):
                continue
            edges[left].append((right, pair_id))
            edges[right].append((left, pair_id))

        visited = set()
        for start in edges:
            if start in visited:
                continue
            members = {start}
            pairs = set()
            pending = [start]
            while pending:
                for other, pair_id in edges[pending.pop()]:
                    pairs.add(pair_id)
                    if other not in members:
                        members.add(other)
                        pending.append(other)
            visited.update(members)
            verdicts = {decisions[pair_id].verdict if pair_id in decisions else "pending" for pair_id in pairs}
            status = next((value for value in ("needs_review", "pending") if value in verdicts), "reviewed")
            group_id = min(pairs)
            groups[group_id] = {"id": group_id, "members": sorted(members), "pairs": sorted(pairs),
                                "status": status, "keep": None, "reason": ""}
        group_for_member = {member: group["id"] for group in groups.values() for member in group["members"]}
        for pair_id, decision in decisions.items():
            if decision.verdict != "distinct":
                continue
            left, right = self.report.pairs[pair_id - 1].question_ids
            group_id = group_for_member.get(left)
            if group_id is not None and group_id == group_for_member.get(right):
                group = groups[group_id]
                if group["keep"] is None:
                    group["pairs"].append(pair_id)
        for group in groups.values():
            group["pairs"].sort()
        return dict(sorted(groups.items()))

    def save_group(self, group_id, members, keep, reason):
        if not isinstance(keep, list) or not all(isinstance(item, str) for item in keep):
            raise ValueError("Choose the questions to keep.")
        if not isinstance(reason, str) or len(reason) > 1000:
            raise ValueError("The note must be no longer than 1,000 characters.")
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            groups = self.groups(self._decisions(db), self._selections(db))
            group = groups.get(group_id)
            if group is None or group["members"] != members:
                raise ValueError("The dispute changed. Open it again before saving.")
            kept = set(keep)
            if not kept or not kept <= set(members):
                raise ValueError("Keep at least one question from this dispute.")
            if not self.report.batch_id.startswith("bank:") and any(
                self.report.questions[item].mongo_id is not None for item in set(members) - kept
            ):
                raise ValueError("Keep the existing bank questions when reviewing a new upload.")
            db.execute("""
                CREATE TABLE IF NOT EXISTS review_group_selections (
                    job_id TEXT NOT NULL, group_id INTEGER NOT NULL, selection TEXT NOT NULL,
                    PRIMARY KEY (job_id, group_id)
                )
            """)
            data = {"members": members, "pairs": group["pairs"], "keep": sorted(kept), "reason": reason.strip()}
            db.execute("""
                INSERT INTO review_group_selections VALUES (?, ?, ?)
                ON CONFLICT (job_id, group_id) DO UPDATE SET selection = excluded.selection
            """, (self.job_id, group_id, json.dumps(data)))

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

    def submit(self, job_id: str, decisions: list[Decision], *, replace: Decision | None = None,
               source: Literal["agent", "manual"] | None = None):
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
            saved_members = {question for item in self._selections(db).values() for question in item["members"]}
            if source:
                db.execute("""
                    CREATE TABLE IF NOT EXISTS review_attribution (
                        job_id TEXT NOT NULL, pair_id INTEGER NOT NULL, source TEXT NOT NULL,
                        verdict TEXT NOT NULL, PRIMARY KEY (job_id, pair_id, source)
                    )
                """)
            for decision in decisions:
                if saved_members.intersection(self.report.pairs[decision.pair_id - 1].question_ids):
                    raise ValueError("This pair is covered by a saved group selection. Edit it in the browser.")
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
                if source:
                    db.execute("""
                        INSERT INTO review_attribution VALUES (?, ?, ?, ?)
                        ON CONFLICT (job_id, pair_id, source) DO UPDATE SET verdict = excluded.verdict
                    """, (*key, source, decision.verdict))
                saved += 1
        return {"job_id": self.job_id, "saved": saved, "already_saved": len(decisions) - saved}
