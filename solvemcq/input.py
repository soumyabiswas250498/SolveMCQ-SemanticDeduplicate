import hashlib
import json
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def required_text(value, field):
    if not isinstance(value, str) or not normalize(value):
        raise ValueError(f"{field} must be a nonempty string")
    return normalize(value)


@dataclass(frozen=True)
class Question:
    id: str
    row: int
    subject: str
    text: str
    answer: str
    choices: tuple
    correct_key: str | None
    record: dict


def load_questions(path: Path, default_subject: str | None):
    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    records = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)
    if isinstance(records, dict):
        records = records.get("questions")
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty JSON array or an object with a questions array")

    batch_id = hashlib.sha256(stable_json([default_subject, records]).encode()).hexdigest()
    questions = []
    for row, record in enumerate(records, 1):
        try:
            if not isinstance(record, dict):
                raise ValueError("expected a JSON object")
            subject = required_text(
                record.get("subjectId", record.get("subject", default_subject)), "subject"
            ).casefold()
            question = required_text(record.get("questionText", record.get("question")), "question")
            choices = {}
            correct_key = None
            answer_fields = {
                field: required_text(record[field], field)
                for field in ("correctAnswer", "answer") if field in record
            }
            if "options" in record:
                options = record["options"]
                if not isinstance(options, list) or len(options) < 2:
                    raise ValueError("options must be an array of at least two {key, text} objects")
                for option in options:
                    if not isinstance(option, dict):
                        raise ValueError("each option must be a {key, text} object")
                    key = required_text(option.get("key"), "option key")
                    if key in choices:
                        raise ValueError(f"duplicate option key {key!r}")
                    choices[key] = required_text(option.get("text"), "option text")
                correct_key = required_text(record.get("correctOptionKey"), "correctOptionKey")
                if correct_key not in choices:
                    raise ValueError("correctOptionKey must match an option key")
                answer = choices[correct_key]
            else:
                answer = required_text(record.get("correctAnswer", record.get("answer")), "answer")
            for field, supplied_answer in answer_fields.items():
                if supplied_answer != answer:
                    raise ValueError(f"{field} disagrees with the resolved answer")
            questions.append(Question(
                str(uuid.uuid5(uuid.NAMESPACE_URL, f"solvemcq:{batch_id}:{row}")),
                row, subject, question, answer, tuple(choices.items()), correct_key, record,
            ))
        except ValueError as error:
            raise ValueError(f"Row {row}: {error}") from error
    return batch_id, questions
