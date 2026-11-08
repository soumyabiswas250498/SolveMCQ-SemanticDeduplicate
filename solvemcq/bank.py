import argparse
import json
import os
import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ApiException

from .input import parse_question, stable_json
from .pipeline import candidate_pair, embedder, find_candidates, index_questions, query_neighbors


def bank_collection(name):
    if not re.fullmatch(r"[a-z0-9_-]{1,60}", name):
        raise ValueError("Bank name must use lowercase letters, digits, hyphens or underscores")
    return f"solvemcq_bank_bge_m3_v1_{name}"


EXCLUDED_STATUSES = ("archived", "inactive")


def read_bank(collection):
    from pymongo import ASCENDING

    projection = dict.fromkeys((
        "_id", "subjectId", "subject", "language", "questionText", "question", "options",
        "correctOptionKey", "correctAnswer", "answer",
    ), 1)
    query = {"deletedAt": None, "operationalStatus": {"$nin": list(EXCLUDED_STATUSES)}}
    return read_bank_documents(collection.find(query, projection).sort("_id", ASCENDING), collection.full_name)


def read_bank_documents(documents, namespace):
    from bson import ObjectId

    questions = []
    metadata = {}
    for row, original in enumerate(documents, 1):
        if not isinstance(original, dict):
            raise ValueError(f"Bank row {row}: expected an object")
        if original.get("deletedAt") is not None or original.get("operationalStatus") in EXCLUDED_STATUSES:
            continue
        document = dict(original)
        mongo_id = document.pop("_id", None)
        try:
            if not isinstance(mongo_id, (ObjectId, str)):
                raise ValueError("expected a string or ObjectId _id")
            source = {"mongo_id": str(mongo_id), "mongo_id_type": type(mongo_id).__name__}
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{stable_json(source)}"))
            if point_id in metadata:
                raise ValueError("duplicate _id in bank")
            if isinstance(document.get("subjectId"), ObjectId):
                document["subjectId"] = str(document["subjectId"])
            question = parse_question(document, row, None, point_id)
        except ValueError as error:
            raise ValueError(f"MongoDB question {mongo_id}: {error}") from error
        questions.append(question)
        metadata[point_id] = source
    return questions, metadata


def sync_bank(client, name, questions, metadata, dimension, encode, batch_size, allow_empty=False, progress=None):
    if not questions and not allow_empty:
        raise ValueError("MongoDB returned zero eligible questions. Verify the source or use --allow-empty.")
    collection = bank_collection(name)

    def payload(item):
        return {
            "subject": item.subject, "language": item.language, "question": item.text, "answer": item.answer,
            **metadata[item.id],
        }

    index_questions(client, collection, questions, dimension, encode, batch_size, payload, progress=progress)
    live = {question.id for question in questions}
    offset = None
    stale = []
    while True:
        points, offset = client.scroll(
            collection, offset=offset, limit=256, with_payload=False, with_vectors=False,
        )
        stale.extend(point.id for point in points if point.id not in live)
        if offset is None:
            break
    for start in range(0, len(stale), 256):
        client.delete(
            collection, points_selector=models.PointIdsList(points=stale[start:start + 256]), wait=True,
        )
    return {"bank": name, "indexed_questions": len(live), "removed_stale_points": len(stale)}


def compare_bank(client, batch_collection, name, questions, top_k, threshold, progress=None):
    collection = bank_collection(name)
    if not client.collection_exists(collection):
        raise ValueError(f"Bank {name!r} is missing. Run solvemcq-bank first.")
    missing_subjects = [
        subject for subject in sorted({item.subject for item in questions})
        if not client.count(
            collection, count_filter=models.Filter(must=[models.FieldCondition(
                key="subject", match=models.MatchValue(value=subject),
            )]),
        ).count
    ]
    known_rows = find_existing_rows(client, collection, questions)
    pairs = []
    bank_questions = {}
    for checked, question in enumerate(questions, 1):
        vectors = client.retrieve(
            batch_collection, [question.id], with_payload=False, with_vectors=True,
        )[0].vector
        hits = query_neighbors(
            client, collection, question, vectors["question"], top_k, threshold, with_payload=True,
        )
        for hit in hits:
            item = hit.payload
            bank_questions[hit.id] = {"input_row": None, **item}
            pairs.append(candidate_pair(
                (question.id, hit.id), question.answer, item["answer"], vectors["answer"], hit,
            ))
        if progress:
            progress(checked, len(questions))
    return pairs, bank_questions, missing_subjects, known_rows


def find_existing_rows(client, collection, questions):
    string_rows = {}
    object_rows = {}
    for question in questions:
        identifier = question.record.get("_id", question.record.get("id"))
        if isinstance(identifier, dict):
            identifier = identifier.get("$oid")
        if isinstance(identifier, str) and identifier:
            string_rows.setdefault(identifier, []).append(question.row)
            if re.fullmatch(r"[0-9a-fA-F]{24}", identifier):
                object_rows.setdefault(identifier.lower(), []).append(question.row)

    known = {}
    offset = None
    while string_rows:
        points, offset = client.scroll(
            collection, offset=offset, limit=256,
            with_payload=["mongo_id", "mongo_id_type"], with_vectors=False,
        )
        for point in points:
            identifier = point.payload["mongo_id"]
            rows = object_rows if point.payload["mongo_id_type"] == "ObjectId" else string_rows
            for row in rows.get(identifier, ()):
                known[row] = identifier
        if offset is None:
            break
    return known


def baseline_report(client, name, questions, metadata, progress=None):
    pairs = find_candidates(client, bank_collection(name), questions, 12, 0.70, progress=progress)
    flagged = {question_id for pair in pairs for question_id in pair["question_ids"]}
    return {
        "schema_version": 2,
        "batch_id": f"bank:{name}",
        "questions": {
            item.id: {
                "input_row": None, "subject": item.subject, "language": item.language, "question": item.text,
                "answer": item.answer, **metadata[item.id],
            }
            for item in questions if item.id in flagged
        },
        "pairs": pairs,
    }


def main():
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError

    parser = argparse.ArgumentParser(description="Sync a MongoDB test question bank to Qdrant.")
    parser.add_argument("--database", required=True)
    parser.add_argument("--collection", default="questions")
    parser.add_argument("--bank", required=True, help="Dedicated Qdrant bank name")
    parser.add_argument("--index-path", type=Path, default=Path(".solvemcq/qdrant"))
    parser.add_argument("--qdrant-url")
    parser.add_argument("--device")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--baseline-report", type=Path, help="Write candidate pairs within the bank")
    args = parser.parse_args()
    try:
        bank_collection(args.bank)
        if args.baseline_report and args.baseline_report.exists():
            raise ValueError("Baseline report already exists. Choose a new path.")
        uri = os.getenv("SOLVEMCQ_MONGO_URI", "mongodb://127.0.0.1:27018")
        with MongoClient(uri, serverSelectionTimeoutMS=5000) as mongo:
            questions, metadata = read_bank(mongo[args.database][args.collection])
        if not questions and not args.allow_empty:
            raise ValueError("MongoDB returned zero eligible questions. Verify the source or use --allow-empty.")
        dimension, encode = embedder(args.device)
        client = (
            QdrantClient(url=args.qdrant_url, api_key=os.getenv("QDRANT_API_KEY"))
            if args.qdrant_url else QdrantClient(path=str(args.index_path))
        )
        try:
            summary = sync_bank(client, args.bank, questions, metadata, dimension, encode, 16, args.allow_empty)
            if args.baseline_report:
                data = baseline_report(client, args.bank, questions, metadata)
                args.baseline_report.parent.mkdir(parents=True, exist_ok=True)
                with args.baseline_report.open("x", encoding="utf-8") as output:
                    json.dump(data, output, ensure_ascii=False, indent=2)
                summary["candidate_pairs"] = len(data["pairs"])
        finally:
            client.close()
    except (ImportError, OSError, ValueError, PyMongoError, ApiException) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
