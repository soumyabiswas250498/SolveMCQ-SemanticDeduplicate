import json
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ReviewServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_review_survives_restart_and_rejects_conflicting_writes(self):
        question = {"subject": "history", "question": "Where?", "answer": "Here"}
        report = {
            "schema_version": 1,
            "batch_id": "sample",
            "questions": {key: {**question, "input_row": row} for row, key in enumerate("abc", 1)},
            "pairs": [{
                "question_ids": list(ids), "question_score": 0.9, "answer_score": 1.0,
                "same_answer_text": True,
            } for ids in ("ab", "ac", "bc")],
        }

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "review_candidates.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            params = StdioServerParameters(command=sys.executable, args=[
                "-m", "solvemcq.review_server", str(path), "--db", str(directory / "review.sqlite"),
            ])

            @asynccontextmanager
            async def connect():
                async with stdio_client(params) as streams, ClientSession(*streams) as session:
                    await session.initialize()
                    yield session

            async with connect() as session:
                tools = {tool.name: tool for tool in (await session.list_tools()).tools}
                self.assertTrue(tools["get_review_batch"].annotations.readOnlyHint)
                self.assertFalse(tools["submit_decisions"].annotations.readOnlyHint)
                result = await session.call_tool("get_review_batch", {"limit": 2})
                self.assertFalse(result.isError)
                batch = result.structuredContent
                self.assertEqual([pair["pair_id"] for pair in batch["candidates"]], [1, 2])
                self.assertTrue((await session.call_tool("get_review_batch", {"limit": 26})).isError)

                async def submit(session, *decisions, job_id=batch["job_id"]):
                    return await session.call_tool("submit_decisions", {
                        "job_id": job_id, "decisions": list(decisions),
                    })

                duplicate = {
                    "pair_id": 1, "verdict": "duplicate", "reason": "Same fact",
                    "keep_question_id": "a",
                }
                deferred = {"pair_id": 2, "verdict": "needs_review", "reason": "Check source"}
                self.assertTrue((await submit(session, duplicate, job_id="wrong")).isError)
                self.assertTrue((await submit(session, {**duplicate, "keep_question_id": "c"})).isError)
                self.assertEqual((await submit(session, duplicate, deferred)).structuredContent["saved"], 2)
                self.assertEqual((await submit(session, duplicate)).structuredContent["already_saved"], 1)

            async with connect() as session:
                pending = (await session.call_tool("get_review_batch")).structuredContent
                self.assertEqual([pair["pair_id"] for pair in pending["candidates"]], [3])
                revisiting = (await session.call_tool(
                    "get_review_batch", {"queue": "needs_review"}
                )).structuredContent
                self.assertEqual(revisiting["candidates"][0]["previous_reason"], "Check source")
                distinct = {"pair_id": 2, "verdict": "distinct", "reason": "Different scope"}
                conflict = {**distinct, "pair_id": 1}
                self.assertTrue((await submit(session, distinct, conflict)).isError)
                summary = (await session.call_tool("get_review_summary")).structuredContent
                self.assertEqual((summary["pending"], summary["duplicates"], summary["needs_review"]), (1, 1, 1))
                self.assertEqual((await submit(session, distinct)).structuredContent["saved"], 1)

            report["pairs"].reverse()
            path.write_text(json.dumps(report), encoding="utf-8")
            async with connect() as session:
                summary = (await session.call_tool("get_review_summary")).structuredContent
                self.assertEqual(summary["pending"], 3)
                self.assertTrue((await submit(session, duplicate)).isError)
