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
                finally:
                    server.shutdown()
                    thread.join()
