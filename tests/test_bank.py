import json
import os
import tempfile
import unittest
import uuid
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from qdrant_client import QdrantClient

from solvemcq.bank import bank_collection, read_bank, sync_bank
from solvemcq.cli import run
from solvemcq.finalize import finalize
from solvemcq.review import Decision, ReviewJob


@unittest.skipUnless(os.getenv("SOLVEMCQ_TEST_MONGO_URI"), "Requires an isolated MongoDB instance")
class BankIntegrationTest(unittest.TestCase):
    def test_sync_match_review_and_export(self):
        from bson import ObjectId
        from pymongo import MongoClient

        mongo = MongoClient(os.environ["SOLVEMCQ_TEST_MONGO_URI"], serverSelectionTimeoutMS=5000)
        self.addCleanup(mongo.close)
        database = mongo[f"solvemcq_test_{uuid.uuid4().hex}"]
        self.addCleanup(mongo.drop_database, database.name)
        collection = database.questions
        subject = ObjectId()
        geography = ObjectId()
        keeper, archived = collection.insert_many([
            {"subjectId": subject, "question": "Where?", "answer": "Here"},
            {"subjectId": geography, "question": "Where?", "answer": "Here"},
        ]).inserted_ids

        def encode(texts, **_):
            return np.array([[0.0, 1.0] if text == "When?" else [1.0, 0.0] for text in texts])

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            index = directory / "qdrant"
            client = QdrantClient(path=str(index))
            try:
                questions, metadata = read_bank(collection)
                sync_bank(client, "test", questions, metadata, 2, encode, 16)
                previous_id = next(item.id for item in questions if metadata[item.id]["mongo_id"] == str(keeper))
                sync_bank(client, "test", questions, metadata, 2, self.fail, 16)
                collection.update_one({"_id": keeper}, {"$set": {"question": "Where exactly?"}})
                collection.update_one({"_id": archived}, {"$set": {"operationalStatus": "archived"}})
                questions, metadata = read_bank(collection)
                summary = sync_bank(client, "test", questions, metadata, 2, encode, 16)
                self.assertEqual(summary["removed_stale_points"], 1)
                updated = client.retrieve(bank_collection("test"), [previous_id])[0]
                self.assertEqual(updated.payload["question"], "Where exactly?")
            finally:
                client.close()

            original = directory / "incoming.json"
            records = [
                {"subjectId": str(subject), "question": "Where was it?", "answer": "Here"},
                {"subjectId": str(subject), "question": "When?", "answer": "Then"},
                {"subjectId": str(geography), "question": "Where was it?", "answer": "Here"},
            ]
            original.write_text(json.dumps(records), encoding="utf-8")
            args = Namespace(
                input=original, output=directory / "phase-one", index_path=index, subject=None,
                bank="test", qdrant_url=None, device="cpu", batch_size=16, top_k=12, threshold=0.7,
            )
            with patch("solvemcq.cli.embedder", return_value=(2, encode)):
                run(args)

            report = args.output / "review_candidates.json"
            db = directory / "reviews.sqlite3"
            job = ReviewJob(report, db)
            self.assertEqual(len(job.report.pairs), 1)
            pair = job.report.pairs[0]
            self.assertIn(previous_id, pair.question_ids)
            incoming_id = next(item for item in pair.question_ids if item != previous_id)
            with self.assertRaisesRegex(ValueError, "must keep the existing"):
                job.submit(job.job_id, [Decision(
                    pair_id=1, verdict="duplicate", reason="Same fact", keep_question_id=incoming_id,
                )])

            provisional = directory / "provisional"
            result = finalize(original, report, db, provisional, None)
            self.assertEqual(result["held_questions"], 1)
            self.assertEqual(
                json.loads((provisional / "import_ready.json").read_text()), records[1:],
            )

            job.submit(job.job_id, [Decision(
                pair_id=1, verdict="duplicate", reason="Same fact", keep_question_id=previous_id,
            )])
            output = directory / "final"
            result = finalize(original, report, db, output, None)
            self.assertTrue(result["complete"])
            self.assertEqual(result["import_ready_questions"], 2)
            removed = json.loads((output / "duplicates.json").read_text())[0]
            self.assertEqual(removed["kept_mongo_id"], str(keeper))
            self.assertEqual(collection.count_documents({}), 2)
