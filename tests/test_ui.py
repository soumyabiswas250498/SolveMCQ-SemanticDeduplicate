import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from solvemcq.ui import LocalServer, LocalWorkflow


class BrowserWorkflowTest(unittest.TestCase):
    def test_upload_review_resume_and_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflow = LocalWorkflow(root)

            def encode(texts, **_):
                return np.array([[0.0, 1.0] if text == "Already saved?" else [1.0, 0.0]
                                 for text in texts])

            workflow.embedding = 2, encode
            with LocalServer(0, workflow) as server:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    def call(path, data=None, *, authorized=True):
                        headers = {"X-SolveMCQ-Token": server.token} if authorized else {}
                        request = Request(f"http://{server.host}{path}",
                                          data=json.dumps(data).encode() if data is not None else None,
                                          headers=headers)
                        with urlopen(request, timeout=5) as response:
                            return json.load(response)

                    with self.assertRaises(HTTPError) as rejected:
                        call("/api/start", {}, authorized=False)
                    self.assertEqual(rejected.exception.code, 403)

                    identifier = "6a5f3cc8dc2d9d345c0d7358"
                    incoming = [
                        {"question": "Already saved?", "answer": "Yes", "id": identifier},
                        {"question": "Where?", "answer": "Here"},
                    ]
                    bank = [{
                        "_id": {"$oid": identifier}, "subjectId": {"$oid": identifier},
                        "questionText": "Existing question?", "correctOptionKey": "A",
                        "options": [{"key": "A", "text": "Here"}, {"key": "B", "text": "There"}],
                    }]
                    state = call("/api/start", {"source": json.dumps(incoming), "bank": json.dumps(bank)})
                    deadline = time.monotonic() + 5
                    while state["status"] == "running" and time.monotonic() < deadline:
                        time.sleep(0.05)
                        state = call("/api/state")
                    self.assertEqual(state["status"], "ready", state)
                    self.assertEqual(state["scan"]["already_in_bank"], 1)
                    job = state["id"]
                    pair = call(f"/api/pair?job={job}")
                    self.assertEqual(len(pair["questions"]), 2)
                    existing = next(item for item in pair["questions"] if "mongo_id" in item)
                    self.assertEqual(existing["options"], bank[0]["options"])
                    call("/api/decide", {"job": job, "pair_id": pair["pair_id"], "verdict": "distinct"})
                    self.assertEqual(LocalWorkflow(root).snapshot()["review"]["distinct"], 1)
                    state = call("/api/export", {"job": job})
                    self.assertEqual(state["export"]["summary"]["already_in_bank_questions"], 1)
                    self.assertEqual(call(f"/api/download?job={job}&file=import_ready.json"), incoming[1:])
                    self.assertEqual(call(f"/api/download?job={job}&file=unique.json"), incoming[1:])
                    self.assertEqual(call(f"/api/download?job={job}&file=drop.json"), incoming[:1])
                finally:
                    server.shutdown()
                    thread.join()

    def test_baseline_agent_handoff_and_archive_preview(self):
        from unittest.mock import patch

        from solvemcq.review import Decision

        with tempfile.TemporaryDirectory() as temporary:
            workflow = LocalWorkflow(Path(temporary))
            workflow.embedding = 2, lambda texts, **_: np.tile([1.0, 0.0], (len(texts), 1))
            records = [
                {"_id": str(index), "subject": "history", "question": f"Version {index}?",
                 "options": [{"key": "A", "text": "Answer"}, {"key": "B", "text": "Other"}],
                 "correctOptionKey": "A"}
                for index in range(1, 4)
            ]

            def wait_for(field, expected):
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    state = workflow.snapshot()
                    if state.get(field) == expected:
                        return state
                    if state["status"] == "error":
                        self.fail(state)
                    time.sleep(0.02)
                self.fail(workflow.snapshot())

            workflow.start({"mode": "baseline", "bank": json.dumps(records)})
            state = wait_for("status", "ready")
            job_id = state["id"]
            self.assertEqual(state["review"]["pending"], 3)

            def agent_decisions(_agent, pairs):
                return [
                    Decision(pair_id=pair["pair_id"], verdict="needs_review", reason="Unclear scope")
                    for pair in pairs
                ]

            with patch("solvemcq.ui.subscription_agent", return_value=object()), patch(
                "solvemcq.ui.review_candidates", side_effect=agent_decisions,
            ):
                workflow.start_agent(job_id)
                state = wait_for("agent_running", False)
            self.assertEqual(state["review"]["pending"], 0)
            self.assertEqual(state["review"]["needs_review"], 3)
            self.assertEqual(state["agent_reviewed_pairs"], 3)
            self.assertGreater(state["review_timing"]["elapsed_seconds"], 0)
            self.assertEqual(state["review_timing"]["remaining_seconds"], 0)
            self.assertEqual(LocalWorkflow(workflow.root).snapshot()["review_timing"], state["review_timing"])

            pair = workflow.pair(job_id, "needs_review")
            self.assertEqual(pair["previous_reason"], "Unclear scope")
            self.assertTrue(all(question["options"] for question in pair["questions"]))
            next_pair = workflow.pair(job_id, "needs_review", after=pair["pair_id"])
            self.assertGreater(next_pair["pair_id"], pair["pair_id"])

            pairs = workflow.job().batch(3, "needs_review")["candidates"]
            for pair in pairs:
                keeper = next((item for item in pair["questions"] if item["mongo_id"] == "1"), None)
                workflow.decide(job_id, {
                    "pair_id": pair["pair_id"], "verdict": "duplicate" if keeper else "distinct",
                    "keep_question_id": keeper["question_id"] if keeper else None,
                })

            state = workflow.export(job_id)
            self.assertEqual(state["export"]["summary"]["proposed_archives"], 0)
            metrics = state["export"]["summary"]["review_metrics"]
            self.assertEqual(metrics["agent_deferred_pairs"], 3)
            self.assertEqual(metrics["agent_resolved_pairs"], 0)
            self.assertEqual(metrics["manual_pairs_resolved"], 3)
            conflict_id = state["export"]["conflicts"][0]["distinct_pair_ids"][0]
            pair = workflow.pair(job_id, "needs_review", conflict=conflict_id)
            workflow.decide(job_id, {
                "pair_id": conflict_id, "conflict": conflict_id, "verdict": "duplicate",
                "keep_question_id": pair["questions"][0]["question_id"],
            })
            state = workflow.export(job_id)
            self.assertEqual(state["export"]["summary"]["proposed_archives"], 2)
            proposed = json.loads(workflow.download(job_id, "proposed_archives.json"))
            self.assertEqual({row["keep_id"] for row in proposed}, {"1"})

    def test_reopen_existing_baseline_with_saved_decisions(self):
        from unittest.mock import patch

        from solvemcq.review import Decision

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflow = LocalWorkflow(root)
            workflow.embedding = 2, lambda texts, **_: np.tile([1.0, 0.0], (len(texts), 1))
            bank = json.dumps([
                {"_id": str(index), "subject": "history", "question": f"Version {index}?", "answer": "Here"}
                for index in range(2)
            ])

            def wait_ready():
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    state = workflow.snapshot()
                    if state["status"] == "ready":
                        return state
                    if state["status"] == "error":
                        self.fail(state)
                    time.sleep(0.02)
                self.fail(workflow.snapshot())

            workflow.start({"mode": "baseline", "bank": bank})
            wait_ready()
            paths = workflow.review_paths()
            original = workflow.job()
            original.submit(original.job_id, [Decision(pair_id=1, verdict="needs_review", reason="Check the source")])
            with patch.object(workflow, "review_paths", return_value=paths):
                workflow.embedding = None
                workflow.start({"mode": "resume", "bank": bank})
                resumed = wait_ready()
                self.assertEqual(resumed["review"]["job_id"], original.job_id)
                self.assertEqual(resumed["review"]["needs_review"], 1)
                workflow.decide(resumed["id"], {"pair_id": 1, "verdict": "distinct"})
            self.assertEqual(original.summary()["distinct"], 1)

    def test_group_keeps_multiple_questions_and_reopens_saved_selection(self):
        from solvemcq.review import Decision

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workflow = LocalWorkflow(root)

            def encode(texts, **_):
                return np.array([[1.0, 0.0] if text.startswith("First") else [0.0, 1.0] for text in texts])

            workflow.embedding = 2, encode
            records = [
                {"_id": str(index), "subject": "history", "question": question, "answer": question}
                for index, question in enumerate(("First A", "First B", "First C", "Second A", "Second B"), 1)
            ]
            records.append({"_id": "6", "deletedAt": "previously archived"})
            workflow.start({"mode": "baseline", "bank": json.dumps(records)})
            deadline = time.monotonic() + 5
            while workflow.snapshot()["status"] == "running" and time.monotonic() < deadline:
                time.sleep(0.02)
            state = workflow.snapshot()
            self.assertEqual(state["status"], "ready", state)
            job_id = state["id"]
            groups = workflow.dispute_groups(job_id)
            self.assertEqual(sorted(item["size"] for item in groups), [2, 3])
            group_id = next(item["id"] for item in groups if item["size"] == 3)
            group = workflow.dispute_groups(job_id, group_id)
            keep = [item["question_id"] for item in group["questions"] if item["mongo_id"] in {"1", "3"}]
            pair = workflow.job().describe_pair(group["pairs"][0])
            job = workflow.job()
            job.submit(job.job_id, [Decision(
                pair_id=pair["pair_id"], verdict="duplicate", reason="Agent matched the pair",
                keep_question_id=pair["question_ids"][0],
            )], source="agent")
            with self.assertRaisesRegex(ValueError, "Keep at least one"):
                workflow.save_dispute(job_id, {"group": group_id, "members": group["members"], "keep": []})
            state = workflow.save_dispute(job_id, {
                "group": group_id, "members": group["members"], "keep": keep, "reason": "Two distinct facts",
            })
            self.assertEqual(state["review"]["resolved_by_selection"], 3)
            self.assertEqual(state["review"]["pending"], 1)
            workflow = LocalWorkflow(root)
            saved = workflow.dispute_groups(job_id, group_id)
            self.assertEqual(saved["keep"], sorted(keep))
            self.assertEqual(saved["reason"], "Two distinct facts")
            state = workflow.export(job_id)
            summary = state["export"]["summary"]
            self.assertEqual(summary["proposed_archives"], 1)
            self.assertEqual(summary["previously_excluded_questions"], 1)
            self.assertEqual((summary["retained_questions"], summary["dropped_questions"]), (4, 2))
            self.assertEqual(summary["review_metrics"]["agent_duplicate_pairs"], 1)
            self.assertEqual(summary["review_metrics"]["manual_groups_resolved"], 1)
            proposals = json.loads(workflow.download(job_id, "proposed_archives.json"))
            self.assertEqual([item["archive_id"] for item in proposals], ["2"])
            self.assertIn(proposals[0]["keep_id"], {"1", "3"})
            self.assertEqual(json.loads(workflow.download(job_id, "unique.json")), records[:1] + records[2:-1])
            self.assertEqual(json.loads(workflow.download(job_id, "drop.json")), records[1:2] + records[-1:])

            workflow.update(bank_snapshot_full=False)
            with self.assertRaisesRegex(ValueError, "Choose the original bank"):
                workflow.export(job_id)
            richer_records = [{**item, "explanation": "Preserve this field"} for item in records]
            workflow.export(job_id, json.dumps(richer_records))
            self.assertEqual(json.loads(workflow.download(job_id, "drop.json")), richer_records[1:2] + richer_records[-1:])
