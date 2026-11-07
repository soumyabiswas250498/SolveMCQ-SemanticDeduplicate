import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from solvemcq.cli import run
from solvemcq.input import load_questions


class PipelineTest(unittest.TestCase):
    def test_exact_semantic_and_cross_subject(self):
        first = {
            "subjectId": "history", "questionText": "Where was the Ghadar Party founded?",
            "correctOptionKey": "B",
            "options": [{"key": "A", "text": "London"}, {"key": "B", "text": "San Francisco"}],
            "chapterId": "one", "explanation": "Preserve the original record.",
        }
        repeat = {**first, "chapterId": "two"}
        paraphrase = {
            "subjectId": "history", "chapterId": "three",
            "question": "গদর দলের প্রতিষ্ঠাস্থল কোথায়?", "answer": "San Francisco",
        }
        conflict = {**paraphrase, "answer": "London"}
        different_subject = {**paraphrase, "subjectId": "geography"}

        def encode(texts, **kwargs):
            self.assertNotIn("Preserve the original record.", texts)
            return np.array([
                [0.0, 1.0] if text == "London" else [1.0, 0.0] for text in texts
            ])

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "input.json"
            input_path.write_text(json.dumps([first, repeat, paraphrase, conflict, different_subject]))
            args = Namespace(
                input=input_path, output=directory / "output", index_path=directory / "index",
                subject=None, qdrant_url=None, device="cpu", batch_size=2, top_k=12, threshold=0.7,
            )
            with patch("solvemcq.cli.embedder", return_value=(2, encode)):
                run(args)

            def read(name):
                return json.loads((args.output / name).read_text())

            self.assertEqual(read("retained.json"), [first, paraphrase, conflict, different_subject])
            self.assertEqual(read("exact_duplicates.json")[0]["input_row"], 2)
            self.assertEqual(read("summary.json")["questions_flagged_for_review"], 3)
            review = read("review_candidates.json")
            self.assertEqual(len(review["pairs"]), 3)
            self.assertEqual(sum(pair["same_answer_text"] for pair in review["pairs"]), 1)
            self.assertEqual(sorted(pair["answer_score"] for pair in review["pairs"]), [0, 0, 1])

            args.output = directory / "retry"
            with patch("solvemcq.cli.embedder", return_value=(2, self.fail)):
                run(args)
            self.assertEqual(read("review_candidates.json"), review)

    def test_rejects_missing_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            path.write_text(json.dumps([{"question": "Question?"}]))
            with self.assertRaisesRegex(ValueError, "Row 1: answer"):
                load_questions(path, "history")
