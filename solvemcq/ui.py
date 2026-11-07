"""Local browser workflow for the SolveMCQ pipeline."""
import argparse
import json
import logging
import os
import subprocess
import secrets
import threading
import uuid
import webbrowser
from argparse import Namespace
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from qdrant_client import QdrantClient

from .agent import review_candidates, subscription_agent
from .bank import baseline_report, read_bank_documents, sync_bank
from .cli import run, write_results
from .finalize import export_baseline, finalize
from .input import load_questions
from .pipeline import embedder
from .review import Decision, ReviewJob


MAX_UPLOAD = 32 * 1024 * 1024
DOWNLOADS = {
    "unique.json", "drop.json",
    "import_ready.json", "duplicates.json", "unresolved.json", "summary.json", "already_in_bank.json",
    "proposed_archives.json",
}


class LocalWorkflow:
    def __init__(self, root, device=None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.lock = threading.RLock()
        self.embedding = None
        self.originals = {}
        self.pause_agent = threading.Event()
        path = self.root / "current.json"
        self.state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "empty"}
        if self.state["status"] == "running":
            self.update(status="error", message="The previous search was interrupted. Please start it again.")

        if self.state.get("agent_running"):
            self.update(agent_running=False, message="Agent review was interrupted. Saved decisions are available.")

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            temp = self.root / "current.tmp"
            temp.write_text(json.dumps(self.state), encoding="utf-8")
            temp.replace(self.root / "current.json")

    def path(self):
        return self.root / self.state["id"]

    def review_paths(self):
        if self.state.get("resume"):
            return Path("output/baseline/review_candidates.json"), Path(".solvemcq/baseline-reviews.sqlite3")
        return self.path() / "scan/review_candidates.json", self.path() / "reviews.sqlite3"

    def job(self):
        return ReviewJob(*self.review_paths(), read_only=True)

    def snapshot(self):
        with self.lock:
            result = {**self.state}
            if result["status"] == "ready":
                result["review"] = self.job().summary()
            return result

    def start(self, data):
        with self.lock:
            if self.state["status"] == "running" or self.state.get("agent_running"):
                raise ValueError("A search or agent review is already running.")
            mode = data.get("mode", "incoming")
            if mode not in ("incoming", "baseline", "resume"):
                raise ValueError("Choose a workflow.")
            source, bank, subject = (data.get(key, "") for key in ("source", "bank", "subject"))
            if not all(isinstance(value, str) for value in (source, bank, subject)) or (mode == "incoming" and not source):
                raise ValueError("Choose an incoming JSON file.")
            database = data.get("database", "solvemcq_test")
            if not isinstance(database, str) or not database.strip():
                raise ValueError("Enter the MongoDB database name.")
            self.state = {
                "id": uuid.uuid4().hex, "status": "running", "message": "Reading your files…",
                "mode": "incoming" if mode == "incoming" else "baseline", "resume": mode == "resume",
            }
            path = self.path()
            path.mkdir()
            if source:
                (path / "incoming.json").write_text(source.removeprefix("\ufeff"), encoding="utf-8")
            if bank:
                (path / "bank.json").write_text(bank.removeprefix("\ufeff"), encoding="utf-8")
            self.originals.clear()
            self.update()
            use_mongo = bool(data.get("use_mongo")) or (mode != "incoming" and not bank)
            threading.Thread(target=self.process, args=(path, subject.strip(), database, use_mongo),
                             daemon=True).start()
            return self.snapshot()

    def process(self, path, subject, database, use_mongo):
        try:
            from bson import json_util

            bank_path = path / "bank.json"
            live_bank = use_mongo and not bank_path.exists()
            if self.state["resume"] and not self.review_paths()[0].is_file():
                raise ValueError("No saved baseline report was found. Choose Clean existing bank to start a scan.")
            if live_bank:
                from pymongo import MongoClient
                from pymongo.errors import PyMongoError

                self.update(message="Reading the existing MongoDB bank…")
                try:
                    uri = os.getenv("SOLVEMCQ_MONGO_URI", "mongodb://127.0.0.1:27018")
                    with MongoClient(uri, serverSelectionTimeoutMS=5000) as mongo:
                        fields = None if self.state["mode"] == "baseline" else {
                            "_id": 1, "subjectId": 1, "subject": 1, "questionText": 1, "question": 1,
                            "options": 1, "correctOptionKey": 1, "correctAnswer": 1, "answer": 1,
                            "operationalStatus": 1, "deletedAt": 1,
                        }
                        docs = list(mongo[database].questions.find({}, fields))
                except PyMongoError as error:
                    raise ValueError("Could not read MongoDB. Start the test database and try again.") from error
                bank_path.write_text(json_util.dumps(docs), encoding="utf-8")
            bank_name = None
            if bank_path.exists():
                docs = json_util.loads(bank_path.read_text(encoding="utf-8"))
                if not isinstance(docs, list):
                    raise ValueError("The question bank must be a MongoDB Extended JSON array.")
                bank_questions, metadata = read_bank_documents(docs, "local-file-bank")
                subjects = {item.subject for item in bank_questions}
                if not subjects:
                    raise ValueError("The question bank contains no eligible questions.")
                if not subject and len(subjects) == 1:
                    subject = next(iter(subjects))
                bank_name = "browser-bank" if live_bank else "local-file"
            else:
                subject = subject or "general"
            subject = subject or None
            is_baseline = self.state["mode"] == "baseline"
            if self.state["resume"]:
                job = ReviewJob(*self.review_paths())
                current = {(meta["mongo_id"], meta["mongo_id_type"]): item
                           for item in bank_questions for meta in [metadata[item.id]]}
                for item in job.report.questions.values():
                    match = current.get((item.mongo_id, item.mongo_id_type))
                    if not match or (match.subject, match.text, match.answer) != (item.subject, item.question, item.answer):
                        raise ValueError("The saved audit differs from the current bank. Start a fresh baseline scan.")
                self.update(status="ready", message="", subject=None, bank_snapshot_full=True,
                            scan={"input_questions": len(bank_questions)})
                return
            if not is_baseline:
                _, incoming = load_questions(path / "incoming.json", subject)
            if bank_name and not is_baseline:
                unknown = sorted({item.subject for item in incoming} - subjects)
                if unknown:
                    raise ValueError("These subjects are missing from the bank: " + ", ".join(unknown))

            self.update(message="Loading the matching model. The first run may take several minutes…")
            if self.embedding is None:
                self.embedding = embedder(self.device)
            index_path = self.root / "qdrant"
            qdrant_url = os.getenv("SOLVEMCQ_QDRANT_URL", "http://127.0.0.1:6333") if live_bank else None
            if bank_name:
                self.update(message="Indexing the existing question bank…")
                client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(path=str(index_path))
                with closing(client):
                    sync_bank(client, bank_name, bank_questions, metadata, *self.embedding, 16)
                    if is_baseline:
                        data = baseline_report(client, bank_name, bank_questions, metadata)
                        write_results(path / "scan", {"review_candidates.json": data})
            if is_baseline:
                ReviewJob(*self.review_paths())
                self.update(status="ready", message="", subject=None, bank_snapshot_full=True,
                            scan={"input_questions": len(bank_questions)})
                return
            self.update(message="Finding similar questions…")
            summary = run(Namespace(
                input=path / "incoming.json", output=path / "scan", subject=subject, index_path=index_path,
                qdrant_url=qdrant_url, device=self.device, bank=bank_name, batch_size=16, top_k=12, threshold=0.70,
            ), embedding=self.embedding)
            ReviewJob(path / "scan/review_candidates.json", path / "reviews.sqlite3")
            self.update(status="ready", message="", subject=subject, scan=summary)
        except (ValueError, OSError, ImportError) as error:
            self.update(status="error", message=str(error))
        except Exception:
            logging.exception("Question search failed")
            self.update(status="error", message="The search failed. See the SolveMCQ window for details.")

    def require_ready(self, job_id):
        if self.state["status"] != "ready" or job_id != self.state["id"]:
            raise ValueError("This batch is no longer open. Refresh the page.")
        return self.path()

    def add_choices(self, path, pairs):
        if not self.originals:
            from bson import json_util

            incoming = path / "incoming.json"
            items = load_questions(incoming, self.state["subject"])[1] if incoming.exists() else []
            self.originals["incoming"] = {item.row: item.record.get("options", []) for item in items}
            bank = path / "bank.json"
            docs = json_util.loads(bank.read_text(encoding="utf-8")) if bank.exists() else []
            self.originals["bank"] = {
                (str(item["_id"]), type(item["_id"]).__name__): item.get("options", [])
                for item in docs if "_id" in item
            }
        for pair in pairs:
            for item in pair["questions"]:
                if "mongo_id" in item:
                    options = self.originals["bank"].get((item["mongo_id"], item["mongo_id_type"]), [])
                else:
                    options = self.originals["incoming"].get(item["input_row"], [])
                item["options"] = options

    def conflict_ids(self):
        export = self.state.get("export") or {}
        return {
            pair_id for group in export.get("conflicts", [])
            for key in ("duplicate_pair_ids", "distinct_pair_ids") for pair_id in group[key]
        }

    def pair(self, job_id, queue, after=0, conflict=None):
        with self.lock:
            path = self.require_ready(job_id)
            job = self.job()
            if conflict is not None:
                if conflict not in self.conflict_ids():
                    raise ValueError("Preview results again to open this conflict.")
                decision = job.saved_decisions()[conflict]
                pairs = [job.describe_pair(conflict, f"Previous decision: {decision.verdict}. {decision.reason}")]
            else:
                pairs = job.batch(1, queue, after=after)["candidates"]
            self.add_choices(path, pairs)
            return pairs[0] if pairs else None

    def dispute_groups(self, job_id, group_id=None):
        with self.lock:
            path = self.require_ready(job_id)
            job = self.job()
            decisions, selections = job.review_state()
            groups = job.groups(decisions, selections)
            if group_id is None:
                return [
                    {"id": group["id"], "status": group["status"], "size": len(group["members"]),
                     "title": job.report.questions[group["members"][0]].question, "pairs": group["pairs"]}
                    for group in groups.values()
                ]
            if group_id not in groups:
                raise ValueError("This dispute changed. Open it again.")
            group = groups[group_id]
            group["questions"] = [
                {"question_id": item, **job.report.questions[item].model_dump(exclude_none=True)}
                for item in group["members"]
            ]
            group["notes"] = [
                {"pair_id": pair_id, "verdict": decisions[pair_id].verdict,
                 "reason": decisions[pair_id].reason,
                 "question_ids": job.report.pairs[pair_id - 1].question_ids}
                for pair_id in group["pairs"] if pair_id in decisions
            ]
            self.add_choices(path, [group])
            return group

    def save_dispute(self, job_id, data):
        with self.lock:
            self.require_ready(job_id)
            if self.state.get("agent_running"):
                raise ValueError("Pause agent review before saving a dispute.")
            self.job().save_group(data.get("group"), data.get("members"), data.get("keep"), data.get("reason", ""))
            self.update(export=None)
            return self.snapshot()

    def start_agent(self, job_id):
        with self.lock:
            path = self.require_ready(job_id)
            if self.state.get("agent_running"):
                raise ValueError("Agent review is already running.")
            self.pause_agent.clear()
            self.update(agent_running=True, export=None, message="Connecting to Codex…")
            threading.Thread(target=self.run_agent, args=(path,), daemon=True).start()
            return self.snapshot()

    def run_agent(self, path):
        message = "Agent review paused. Decisions already saved are available."
        try:
            agent = subscription_agent()
            job = self.job()
            while not self.pause_agent.is_set():
                candidates = job.batch(15, "pending")["candidates"]
                if not candidates:
                    message = "Agent review finished. Open Needs your decision for any remaining pairs."
                    break
                with self.lock:
                    self.add_choices(path, candidates)
                self.update(message=f"Agent review running. {job.summary()['pending']} pairs still to review…")
                decisions = review_candidates(agent, candidates)
                job.submit(job.job_id, decisions)
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            message = f"Agent review stopped. {error}"
        except Exception:
            logging.exception("Agent review failed")
            message = "Agent review stopped. Saved decisions are available. See the app window for details."
        finally:
            self.update(agent_running=False, message=message, export=None)

    def stop_agent(self, job_id):
        with self.lock:
            self.require_ready(job_id)
            if self.state.get("agent_running"):
                self.pause_agent.set()
                self.update(message="Pausing after the current batch…")
            return self.snapshot()

    def decide(self, job_id, data):
        with self.lock:
            self.require_ready(job_id)
            if self.state.get("agent_running"):
                raise ValueError("Pause agent review before making manual decisions.")
            job = self.job()
            verdict = data.get("verdict")
            labels = {"duplicate": "Marked as the same question", "distinct": "Marked as different questions",
                      "needs_review": "Deferred for later review"}
            if verdict not in labels:
                raise ValueError("Choose a review decision.")
            decision = Decision.model_validate({
                "pair_id": data.get("pair_id"), "verdict": verdict,
                "reason": data.get("reason") or labels[verdict],
                "keep_question_id": data.get("keep_question_id"),
            })
            previous = None
            if data.get("conflict") is not None:
                if data["conflict"] != decision.pair_id or decision.pair_id not in self.conflict_ids():
                    raise ValueError("Preview results again to revise this conflict.")
                previous = job.saved_decisions()[decision.pair_id]
            job.submit(job.job_id, [decision], replace=previous)
            self.update(export=None)
            return self.snapshot()

    def export(self, job_id, original_bank=None):
        with self.lock:
            path = self.require_ready(job_id)
            if self.state.get("agent_running"):
                raise ValueError("Pause agent review before preparing results.")
            name = f"export-{uuid.uuid4().hex}"
            if self.state.get("mode") == "baseline":
                bank_source = (path / "bank.json").read_text(encoding="utf-8")
                if not self.state.get("bank_snapshot_full"):
                    if not isinstance(original_bank, str) or not original_bank:
                        raise ValueError("Choose the original bank JSON to preserve all fields in these downloads.")
                    self.validate_original_bank(bank_source, original_bank)
                    bank_source = original_bank.removeprefix("\ufeff")
                summary = export_baseline(self.job(), path / name, bank_source)
                if not self.state.get("bank_snapshot_full"):
                    recovered = path / "bank-recovered.json"
                    recovered.write_text(bank_source, encoding="utf-8")
                    recovered.replace(path / "bank.json")
                    self.update(bank_snapshot_full=True)
            else:
                report, db = self.review_paths()
                summary = finalize(
                    path / "incoming.json", report, db, path / name, self.state["subject"], reviewed_only=True,
                )
            export = {"directory": name, "summary": summary}
            unresolved = json.loads((path / name / "unresolved.json").read_text(encoding="utf-8"))
            export["conflicts"] = unresolved["blocked_duplicate_groups"]
            if self.state.get("mode") == "baseline":
                export["preview"] = json.loads((path / name / "proposed_archives.json").read_text(encoding="utf-8"))
            self.update(export=export)
            return self.snapshot()

    @staticmethod
    def validate_original_bank(snapshot, original):
        from bson import json_util

        records = json_util.loads(original)
        if not isinstance(records, list) or not all(isinstance(item, dict) and "_id" in item for item in records):
            raise ValueError("Choose a MongoDB JSON array with original record IDs.")
        indexed = {item["_id"]: item for item in records}
        saved = json_util.loads(snapshot)
        if len(indexed) != len(records) or len(saved) != len(records) or any(
            item["_id"] not in indexed or any(indexed[item["_id"]].get(key) != value for key, value in item.items())
            for item in saved
        ):
            raise ValueError("The original bank differs from the snapshot used for this review.")

    def download(self, job_id, filename):
        with self.lock:
            path = self.require_ready(job_id)
            export = self.state.get("export")
            if filename not in DOWNLOADS or not export:
                raise ValueError("Generate the results first.")
            return (path / export["directory"] / filename).read_bytes()


class LocalServer(ThreadingHTTPServer):
    def __init__(self, port, workflow):
        super().__init__(("127.0.0.1", port), LocalHandler)
        self.workflow = workflow
        self.host = f"127.0.0.1:{self.server_port}"
        self.token = secrets.token_urlsafe(32)


class LocalHandler(BaseHTTPRequestHandler):
    def send_content(self, status, content, mime, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                         "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, status, value):
        self.send_content(status, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch()

    def dispatch(self):
        if self.headers.get("Host") != self.server.host:
            self.send_json(403, {"error": "Open SolveMCQ using its local address."})
            return
        if self.command == "POST" and self.headers.get("X-SolveMCQ-Token") != self.server.token:
            self.send_json(403, {"error": "Refresh the SolveMCQ page."})
            return
        url = urlsplit(self.path)
        try:
            self.route(url.path, {key: values[0] for key, values in parse_qs(url.query).items()})
        except (ValueError, OSError) as error:
            self.send_json(400, {"error": str(error)})
        except Exception:
            logging.exception("Browser request failed")
            self.send_json(500, {"error": "Something went wrong. See the SolveMCQ window for details."})

    def route(self, path, query):
        workflow = self.server.workflow
        if self.command == "GET":
            if path == "/":
                page = Path(__file__).with_name("web").joinpath("index.html").read_text(encoding="utf-8")
                self.send_content(200, page.replace("__LOCAL_TOKEN__", self.server.token).encode(),
                                  "text/html; charset=utf-8")
            elif path == "/api/state":
                self.send_json(200, workflow.snapshot())
            elif path == "/api/pair":
                queue = query.get("queue", "pending")
                if queue not in ("pending", "needs_review"):
                    raise ValueError("Unknown review list.")
                conflict = int(query["conflict"]) if "conflict" in query else None
                self.send_json(200, workflow.pair(query.get("job"), queue, int(query.get("after", "0")), conflict))
            elif path == "/api/disputes":
                group_id = int(query["group"]) if "group" in query else None
                self.send_json(200, workflow.dispute_groups(query.get("job"), group_id))
            elif path == "/api/download":
                filename = query.get("file")
                self.send_content(200, workflow.download(query.get("job"), filename),
                                  "application/json; charset=utf-8", filename)
            else:
                self.send_json(404, {"error": "Page not found."})
            return

        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_UPLOAD:
            raise ValueError("Choose JSON files smaller than 32 MB combined.")
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        if path == "/api/start":
            result = workflow.start(data)
        elif path == "/api/decide":
            result = workflow.decide(data.get("job"), data)
        elif path == "/api/save-dispute":
            result = workflow.save_dispute(data.get("job"), data)
        elif path == "/api/agent":
            result = workflow.start_agent(data.get("job"))
        elif path == "/api/pause":
            result = workflow.stop_agent(data.get("job"))
        elif path == "/api/export":
            result = workflow.export(data.get("job"), data.get("original_bank"))
        else:
            self.send_json(404, {"error": "Page not found."})
            return
        self.send_json(200, result)


def main():
    parser = argparse.ArgumentParser(description="Open the local SolveMCQ browser app.")
    parser.add_argument("--data-dir", type=Path, default=Path(".solvemcq/browser"))
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    with LocalServer(args.port, LocalWorkflow(args.data_dir, args.device)) as server:
        address = f"http://{server.host}"
        print(f"SolveMCQ is open at {address}. Keep this window open.", flush=True)
        if not args.no_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
