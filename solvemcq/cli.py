import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from qdrant_client import QdrantClient

from .bank import compare_bank
from .input import load_questions
from .pipeline import MODEL, embedder, find_candidates, index_questions, split_exact


def run(args):
    if args.output.exists():
        raise ValueError("Output directory already exists. Choose a new output path.")
    if min(args.top_k, args.batch_size) < 1 or not -1 <= args.threshold <= 1:
        raise ValueError("Expected positive batch size and top-k, and a threshold between -1 and 1")
    batch_id, questions = load_questions(args.input, args.subject)
    retained, duplicates = split_exact(questions)
    collection = f"solvemcq_bge_m3_v1_{batch_id}"
    dimension, encode = embedder(args.device)
    client = (
        QdrantClient(url=args.qdrant_url, api_key=os.getenv("QDRANT_API_KEY"))
        if args.qdrant_url else QdrantClient(path=str(args.index_path))
    )
    bank_name = getattr(args, "bank", None)
    bank_questions = {}
    try:
        index_questions(client, collection, retained, dimension, encode, args.batch_size)
        pairs = find_candidates(client, collection, retained, args.top_k, args.threshold)
        if bank_name:
            cross_pairs, bank_questions, missing_subjects = compare_bank(
                client, collection, bank_name, retained, args.top_k, args.threshold,
            )
            pairs.extend(cross_pairs)
            pairs.sort(key=lambda pair: (-pair["question_score"], pair["question_ids"]))
    finally:
        client.close()

    review_ids = {
        question_id for pair in pairs for question_id in pair["question_ids"]
        if question_id not in bank_questions
    }
    summary = {
        "input_questions": len(questions),
        "exact_duplicates": len(duplicates),
        "retained_questions": len(retained),
        "questions_flagged_for_review": len(review_ids),
        "candidate_pairs": len(pairs),
        "retained_without_candidate": len(retained) - len(review_ids),
    }
    if bank_name:
        summary.update({
            "bank": bank_name,
            "bank_candidate_pairs": len(cross_pairs),
            "bank_subjects_without_records": missing_subjects,
        })
    results = {
        "retained.json": [item.record for item in retained],
        "exact_duplicates.json": duplicates,
        "review_candidates.json": {
            "schema_version": 2 if bank_name else 1,
            "batch_id": batch_id,
            "questions": bank_questions | {
                item.id: {
                    "input_row": item.row,
                    "subject": item.subject,
                    "question": item.text,
                    "answer": item.answer,
                }
                for item in retained if item.id in review_ids
            },
            "pairs": pairs,
        },
        "summary.json": {
            "batch_id": batch_id, "embedding_model": MODEL, "collection": collection,
            "question_threshold": args.threshold, "top_k": args.top_k, **summary,
        },
    }
    write_results(args.output, results)
    print(json.dumps(summary, indent=2))


def write_results(output, results):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        staging = Path(temporary) / "results"
        staging.mkdir()
        for name, data in results.items():
            (staging / name).write_text(
                json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description="Find SolveMCQ duplicates in one JSON batch.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject", help="Default subject for rows without subjectId or subject")
    parser.add_argument("--index-path", type=Path, default=Path(".solvemcq/qdrant"))
    parser.add_argument("--qdrant-url", help="Use an existing Qdrant server instead of embedded storage")
    parser.add_argument("--bank", help="Compare against a synced MongoDB bank as well")
    parser.add_argument("--device", help="Embedding device, for example cpu, mps or cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()
    try:
        run(args)
    except (ImportError, OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
