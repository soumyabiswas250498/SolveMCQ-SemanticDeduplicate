import argparse
import json
import sqlite3
from collections import defaultdict
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

from .cli import write_results
from .input import load_questions
from .pipeline import split_exact
from .review import ReviewJob


def finalize(input_path, report_path, db_path, output, subject, *, reviewed_only=False):
    if output.exists():
        raise ValueError("Output directory already exists. Choose a new output path.")

    batch_id, questions = load_questions(input_path, subject)
    retained, exact = split_exact(questions)
    by_id = {question.id: question for question in retained}
    job = ReviewJob(report_path, db_path, read_only=True)
    if job.report.batch_id != batch_id:
        raise ValueError("Candidate report does not match the input and subject used in Phase 1")
    existing = {
        question_id: item for question_id, item in job.report.questions.items()
        if item.mongo_id is not None
    }
    for question_id, item in job.report.questions.items():
        if question_id in existing:
            continue
        original = by_id.get(question_id)
        if original is None or (item.input_row, item.subject, item.question, item.answer) != (
            original.row, original.subject, original.text, original.answer,
        ):
            raise ValueError(f"Candidate question does not match the original input: {question_id}")

    def describe(question_id):
        if question_id in existing:
            return {"question_id": question_id, **existing[question_id].model_dump(exclude_none=True)}
        item = by_id[question_id]
        return {"question_id": question_id, "input_row": item.row, "record": item.record}

    def keeper_fields(question_id):
        if question_id in existing:
            item = existing[question_id]
            return {"kept_id": question_id, "kept_mongo_id": item.mongo_id,
                    "kept_mongo_id_type": item.mongo_id_type, "kept_input_row": None}
        return {"kept_id": question_id, "kept_input_row": by_id[question_id].row}

    removed_to_keeper, evidence, unresolved, conflicts, held = reconcile(job, describe, existing)
    semantic = [
        {"type": "semantic", "removed_id": item, "input_row": by_id[item].row,
         "record": by_id[item].record, **keeper_fields(keeper), "duplicate_pair_ids": evidence[item]}
        for item, keeper in removed_to_keeper.items()
    ]
    duplicates = semantic.copy()
    for item in exact:
        keeper = removed_to_keeper.get(item["kept_id"], item["kept_id"])
        duplicates.append({
            **item, "type": "exact", **keeper_fields(keeper),
        })
    duplicates.sort(key=lambda item: item["input_row"])
    unique = [item.record for item in retained if item.id not in removed_to_keeper]
    summary = {
        "job_id": job.job_id,
        "input_questions": len(questions),
        "retained_questions": len(unique),
        "exact_duplicates_removed": len(exact),
        "semantic_duplicates_removed": len(semantic),
        "pending_pairs": sum(item["status"] == "pending" for item in unresolved),
        "deferred_pairs": sum(item["status"] == "needs_review" for item in unresolved),
        "blocked_duplicate_groups": len(conflicts),
        "complete": not unresolved and not conflicts,
    }
    results = {
        "unique.json": unique, "duplicates.json": duplicates,
        "unresolved.json": {"pairs": unresolved, "blocked_duplicate_groups": conflicts},
        "summary.json": summary,
    }
    if job.report.schema_version == 2 or reviewed_only:
        existing_ids = job.report.existing_ids or {}
        if not existing_ids.keys() <= {item.row for item in retained}:
            raise ValueError("Candidate report contains an unknown existing input row")
        ready = [
            item.record for item in retained
            if item.id not in removed_to_keeper and item.id not in held and item.row not in existing_ids
        ]
        results["import_ready.json"] = ready
        results["already_in_bank.json"] = [
            {"input_row": item.row, "mongo_id": existing_ids[item.row], "record": item.record}
            for item in retained if item.row in existing_ids
        ]
        summary["already_in_bank_questions"] = len(existing_ids)
        summary["import_ready_questions"] = len(ready)
        summary["held_questions"] = len(unique) - len(ready)
    write_results(output, results)
    return summary


def reconcile(job, describe, protected_ids=()):
    decisions, selections = job.review_state()
    covered = {pair_id for item in selections.values() for pair_id in item["pairs"]}
    if any(pair_id > len(job.report.pairs) for pair_id in decisions):
        raise ValueError("Review database contains an unknown pair ID for this report")

    neighbors = defaultdict(set)
    keep_targets = defaultdict(set)
    duplicate_pairs = defaultdict(set)
    distinct_pairs = defaultdict(dict)
    unresolved = []
    blocked = defaultdict(set)

    for pair_id, pair in enumerate(job.report.pairs, 1):
        if pair_id in covered:
            continue
        left, right = pair.question_ids
        decision = decisions.get(pair_id)
        if decision is None or decision.verdict == "needs_review":
            for question_id in pair.question_ids:
                blocked[question_id].add(pair_id)
            unresolved.append({
                "pair_id": pair_id,
                "status": decision.verdict if decision else "pending",
                "reason": decision.reason if decision else None,
                "questions": [describe(question_id) for question_id in pair.question_ids],
            })
        elif decision.verdict == "distinct":
            distinct_pairs[left][right] = pair_id
        else:
            keeper = decision.keep_question_id
            if keeper not in pair.question_ids:
                raise ValueError(f"Invalid keeper for pair {pair_id}")
            removed = right if keeper == left else left
            keep_targets[removed].add(keeper)
            neighbors[left].add(right)
            neighbors[right].add(left)
            duplicate_pairs[left].add(pair_id)

    removed_to_keeper = {}
    evidence = {}
    conflicts = []
    visited = set()
    held = set(blocked)
    for start in neighbors:
        if start in visited:
            continue
        component = {start}
        todo = [start]
        while todo:
            for neighbor in neighbors[todo.pop()] - component:
                component.add(neighbor)
                todo.append(neighbor)
        visited.update(component)

        pending_ids = sorted({pair_id for item in component for pair_id in blocked[item]})
        distinct_ids = sorted(
            pair_id for item in component for other, pair_id in distinct_pairs[item].items()
            if other in component
        )
        pair_ids = sorted({pair_id for item in component for pair_id in duplicate_pairs[item]})
        keepers = [item for item in component if not keep_targets[item]]
        reasons = []
        existing_members = component.intersection(protected_ids)
        if existing_members and (len(existing_members) != 1 or keepers != list(existing_members)):
            reasons.append("Group does not resolve to exactly one existing MongoDB keeper")
        if pending_ids:
            reasons.append("Touches an unresolved pair")
        if distinct_ids:
            reasons.append("Contains a pair marked distinct")
        if len(keepers) != 1:
            reasons.append("Keeper decisions do not identify one retained question")
        try:
            tuple(TopologicalSorter({item: keep_targets[item] for item in component}).static_order())
        except CycleError:
            reasons.append("Keeper decisions form a cycle")

        if reasons:
            held.update(component)
            conflicts.append({
                "reasons": reasons, "duplicate_pair_ids": pair_ids,
                "distinct_pair_ids": distinct_ids, "unresolved_pair_ids": pending_ids,
                "questions": [
                    describe(item) for item in sorted(component)
                ],
            })
            continue

        keeper = keepers[0]
        for item in component - {keeper}:
            removed_to_keeper[item] = keeper
            evidence[item] = pair_ids
    for selection in selections.values():
        keep = selection["keep"]
        scores = {}
        for pair_id in selection["pairs"]:
            pair = job.report.pairs[pair_id - 1]
            scores[frozenset(pair.question_ids)] = pair.question_score
        for item in set(selection["members"]) - set(keep):
            keeper = max(keep, key=lambda other: scores.get(frozenset((item, other)), -2))
            removed_to_keeper[item] = keeper
            evidence[item] = selection["pairs"]
    return removed_to_keeper, evidence, unresolved, conflicts, held


def export_baseline(job, output):
    questions = job.report.questions
    if not job.report.batch_id.startswith("bank:") or any(not item.mongo_id for item in questions.values()):
        raise ValueError("Choose an existing-bank baseline report.")

    def describe(question_id):
        return {"question_id": question_id, **questions[question_id].model_dump(exclude_none=True)}

    removed, evidence, unresolved, conflicts, _ = reconcile(job, describe)
    proposals = [
        {"archive_id": questions[item].mongo_id, "archive_id_type": questions[item].mongo_id_type,
         "keep_id": questions[keeper].mongo_id, "keep_id_type": questions[keeper].mongo_id_type,
         "archive_question": questions[item].question, "keep_question": questions[keeper].question,
         "duplicate_pair_ids": evidence[item]}
        for item, keeper in sorted(removed.items(), key=lambda pair: questions[pair[0]].mongo_id)
    ]
    summary = {
        "proposed_archives": len(proposals),
        "pending_pairs": sum(item["status"] == "pending" for item in unresolved),
        "deferred_pairs": sum(item["status"] == "needs_review" for item in unresolved),
        "blocked_duplicate_groups": len(conflicts),
        "complete": not unresolved and not conflicts,
    }
    write_results(output, {
        "proposed_archives.json": proposals, "summary.json": summary,
        "unresolved.json": {"pairs": unresolved, "blocked_duplicate_groups": conflicts},
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export SolveMCQ questions from saved review decisions.")
    parser.add_argument("input", type=Path, help="Original JSON used in Phase 1")
    parser.add_argument("report", type=Path, help="Phase 1 review_candidates.json")
    parser.add_argument("--db", type=Path, default=Path(".solvemcq/reviews.sqlite3"))
    parser.add_argument("--subject", help="Same subject fallback used in Phase 1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = finalize(args.input, args.report, args.db, args.output, args.subject)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
