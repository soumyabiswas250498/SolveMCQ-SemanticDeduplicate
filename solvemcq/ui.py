"""Local browser workflow for the SolveMCQ pipeline."""
import argparse
import json
import logging
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

from .bank import read_bank_documents, sync_bank
from .cli import run
from .finalize import finalize
from .input import load_questions
from .pipeline import embedder
from .review import Decision, ReviewJob


MAX_UPLOAD = 32 * 1024 * 1024
DOWNLOADS = {
    "import_ready.json", "duplicates.json", "unresolved.json", "summary.json", "already_in_bank.json",
}


class LocalWorkflow:
    def __init__(self, root, device=None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.lock = threading.RLock()
        self.embedding = None
        self.originals = {}
        path = self.root / "current.json"
        self.state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "empty"}
        if self.state["status"] == "running":
            self.update(status="error", message="The previous search was interrupted. Please start it again.")

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            temp = self.root / "current.tmp"
            temp.write_text(json.dumps(self.state), encoding="utf-8")
            temp.replace(self.root / "current.json")

    def path(self):
        return self.root / self.state["id"]

    def job(self):
        path = self.path()
        return ReviewJob(path / "scan/review_candidates.json", path / "reviews.sqlite3", read_only=True)

    def snapshot(self):
        with self.lock:
            result = {**self.state}
            if result["status"] == "ready":
                result["review"] = self.job().summary()
            return result

    def start(self, data):
        with self.lock:
            if self.state["status"] == "running":
                raise ValueError("A search is already running.")
            source, bank, subject = (data.get(key, "") for key in ("source", "bank", "subject"))
            if not all(isinstance(value, str) for value in (source, bank, subject)) or not source:
                raise ValueError("Choose an incoming JSON file.")
            self.state = {"id": uuid.uuid4().hex, "status": "running", "message": "Reading your files…"}
            path = self.path()
            path.mkdir()
            (path / "incoming.json").write_text(source.removeprefix("\ufeff"), encoding="utf-8")
            if bank:
                (path / "bank.json").write_text(bank.removeprefix("\ufeff"), encoding="utf-8")
            self.originals.clear()
            self.update()
            threading.Thread(target=self.process, args=(path, subject.strip()), daemon=True).start()
            return self.snapshot()

    def process(self, path, subject):
        try:
            bank_path = path / "bank.json"
            bank_name = None
            if bank_path.exists():
                from bson import json_util

                docs = json_util.loads(bank_path.read_text(encoding="utf-8"))
                if not isinstance(docs, list):
                    raise ValueError("The question bank must be a MongoDB Extended JSON array.")
                bank_questions, metadata = read_bank_documents(docs, "local-file-bank")
                subjects = {item.subject for item in bank_questions}
                if not subjects:
                    raise ValueError("The question bank contains no eligible questions.")
                if not subject and len(subjects) == 1:
                    subject = next(iter(subjects))
                bank_name = "local-file"
            else:
                subject = subject or "general"
            subject = subject or None
            _, incoming = load_questions(path / "incoming.json", subject)
            if bank_name:
                unknown = sorted({item.subject for item in incoming} - subjects)
                if unknown:
                    raise ValueError("These subjects are missing from the bank: " + ", ".join(unknown))

            self.update(message="Loading the matching model. The first run may take several minutes…")
            if self.embedding is None:
                self.embedding = embedder(self.device)
            index_path = self.root / "qdrant"
            if bank_name:
                self.update(message="Indexing the existing question bank…")
                with closing(QdrantClient(path=str(index_path))) as client:
                    sync_bank(client, bank_name, bank_questions, metadata, *self.embedding, 16)
            self.update(message="Finding similar questions…")
            summary = run(Namespace(
                input=path / "incoming.json", output=path / "scan", subject=subject, index_path=index_path,
                qdrant_url=None, device=self.device, bank=bank_name, batch_size=16, top_k=12, threshold=0.70,
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

    def pair(self, job_id, queue):
        with self.lock:
            path = self.require_ready(job_id)
            job = self.job()
            result = job.batch(1, queue)
            if not result["candidates"]:
                return None
            pair = result["candidates"][0]
            if not self.originals:
                from bson import json_util

                _, items = load_questions(path / "incoming.json", self.state["subject"])
                self.originals["incoming"] = {item.row: item.record.get("options", []) for item in items}
                bank = path / "bank.json"
                docs = json_util.loads(bank.read_text(encoding="utf-8")) if bank.exists() else []
                self.originals["bank"] = {
                    (str(item["_id"]), type(item["_id"]).__name__): item.get("options", [])
                    for item in docs if "_id" in item
                }
            for item in pair["questions"]:
                if "mongo_id" in item:
                    options = self.originals["bank"].get((item["mongo_id"], item["mongo_id_type"]), [])
                else:
                    options = self.originals["incoming"].get(item["input_row"], [])
                item["options"] = options
            return pair

    def decide(self, job_id, data):
        with self.lock:
            self.require_ready(job_id)
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
            job.submit(job.job_id, [decision])
            self.update(export=None)
            return self.snapshot()

    def export(self, job_id):
        with self.lock:
            path = self.require_ready(job_id)
            name = f"export-{uuid.uuid4().hex}"
            summary = finalize(
                path / "incoming.json", path / "scan/review_candidates.json", path / "reviews.sqlite3",
                path / name, self.state["subject"], reviewed_only=True,
            )
            self.update(export={"directory": name, "summary": summary})
            return self.snapshot()

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
                self.send_json(200, workflow.pair(query.get("job"), queue))
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
        elif path == "/api/export":
            result = workflow.export(data.get("job"))
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
