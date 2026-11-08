import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from solvemcq.finalize import finalize
from solvemcq.input import load_questions
from solvemcq.review import Decision, ReviewJob


class FinalizeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.input = self.directory / "input.json"
        self.report = self.directory / "review_candidates.json"
        self.db = self.directory / "reviews.sqlite3"
        self.output = self.directory / "output"
        self.records = [
            {"question": question, "answer": "answer", "id": index}
            for index, question in enumerate(("A?", "B?", "C?", "A?", "D?", "E?"), 1)
        ]
        self.input.write_text(json.dumps(self.records), encoding="utf-8")

    def job(self, pairs):
        batch_id, questions = load_questions(self.input, "history")
        by_row = {item.row: item for item in questions}
        self.ids = {row: item.id for row, item in by_row.items()}
        self.report.write_text(json.dumps({
            "schema_version": 1,
            "batch_id": batch_id,
            "questions": {
                self.ids[row]: {
                    "input_row": row, "subject": "history",
                    "question": by_row[row].text, "answer": by_row[row].answer,
                    "language": by_row[row].language,
                }
                for row in sorted({row for pair in pairs for row in pair})
            },
            "pairs": [
                {"question_ids": [self.ids[left], self.ids[right]], "question_score": 0.9,
                 "answer_score": 1.0, "same_answer_text": True}
                for left, right in pairs
            ],
        }), encoding="utf-8")
        return ReviewJob(self.report, self.db)

    def save(self, job, pair_id, verdict, keeper=None):
        job.submit(job.job_id, [Decision(
            pair_id=pair_id, verdict=verdict, reason="Test decision",
            keep_question_id=self.ids[keeper] if keeper is not None else None,
        )])

    def export(self, subject="history"):
        return finalize(self.input, self.report, self.db, self.output, subject)

    def read(self, name):
        return json.loads((self.output / name).read_text(encoding="utf-8"))

    def test_follows_keeper_chain_and_repoints_exact_duplicates(self):
        job = self.job([(1, 2), (2, 3)])
        self.save(job, 1, "duplicate", 2)
        self.save(job, 2, "duplicate", 3)
        summary = self.export()
        self.assertTrue(summary["complete"])
        self.assertEqual(self.read("unique.json"), [self.records[i] for i in (2, 4, 5)])
        duplicates = self.read("duplicates.json")
        self.assertEqual([item["input_row"] for item in duplicates], [1, 2, 4])
        self.assertEqual([item["kept_input_row"] for item in duplicates], [3, 3, 3])

    def test_preserves_conflicted_and_deferred_groups(self):
        job = self.job([(1, 2), (2, 3), (1, 3), (3, 5), (5, 6)])
        self.save(job, 1, "duplicate", 1)
        self.save(job, 2, "duplicate", 2)
        self.save(job, 3, "distinct")
        self.save(job, 4, "needs_review")
        self.save(job, 5, "duplicate", 5)
        summary = self.export()
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["blocked_duplicate_groups"], 2)
        self.assertEqual(summary["semantic_duplicates_removed"], 0)
        unresolved = self.read("unresolved.json")
        self.assertEqual(unresolved["pairs"][0]["questions"][1]["record"], self.records[4])
        self.assertEqual(unresolved["blocked_duplicate_groups"][0]["distinct_pair_ids"], [3])

    def test_preserves_groups_with_multiple_keepers_or_cycles(self):
        for pairs, keepers in (
            ([(1, 2), (2, 3)], [1, 3]),
            ([(1, 2), (2, 3), (1, 3)], [1, 2, 3]),
        ):
            with self.subTest(keepers=keepers):
                job = self.job(pairs)
                for pair_id, keeper in enumerate(keepers, 1):
                    self.save(job, pair_id, "duplicate", keeper)
                self.output = self.directory / f"output-{len(pairs)}"
                summary = self.export()
                self.assertEqual(summary["semantic_duplicates_removed"], 0)
                self.assertEqual(summary["blocked_duplicate_groups"], 1)

    def test_untagged_record_cannot_merge_two_known_language_versions(self):
        self.records[0]["language"] = "bn"
        self.records[2]["language"] = "en"
        self.input.write_text(json.dumps(self.records), encoding="utf-8")
        job = self.job([(1, 2), (2, 3)])
        self.save(job, 1, "duplicate", 2)
        self.save(job, 2, "duplicate", 2)
        summary = self.export()
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["semantic_duplicates_removed"], 0)
        self.assertIn("This group contains different language versions",
                      self.read("unresolved.json")["blocked_duplicate_groups"][0]["reasons"])

    def test_rejects_wrong_input_or_missing_database(self):
        self.job([(1, 2)])
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.export(subject="geography")
        self.db = self.directory / "missing.sqlite3"
        with self.assertRaisesRegex(sqlite3.OperationalError, "unable to open database file"):
            self.export()
        self.assertFalse(self.db.exists())
        self.assertFalse(self.output.exists())
