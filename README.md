# SolveMCQ deduplication pipeline

Phase 1 indexes a JSON batch locally and reports exact repeats and semantic review candidates. The retained records preserve the original options, explanations, and metadata.

## Run locally

Use Python 3.11 or later. The first model load downloads BGE-M3 from Hugging Face. Inference runs locally.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[embeddings]'
solvemcq examples/mcq.json --output output/example
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. Add `--device cpu`, `--device mps` or `--device cuda` to select a device explicitly.

Embedded Qdrant persists under `.solvemcq/qdrant`. Use `--qdrant-url http://localhost:6333` for an existing server and set `QDRANT_API_KEY` if it requires a key. Use a dedicated index for this pipeline. Identical batches reuse their stored embeddings. Each distinct batch has its own collection and is searched independently.

## Input

Accepts a JSON array or an object containing a `questions` array. Each row needs:

- `subjectId` or `subject` as a string, or a CLI fallback such as `--subject history`. Use consistent subject IDs across the input.
- Plain-text `questionText` or `question`.
- Plain-text `correctAnswer` or `answer`, or an `options` array of `{key, text}` objects and a matching `correctOptionKey`. When both forms are supplied, they must agree.

The CLI rejects malformed JSON, including trailing commas, and reports the line and column. Invalid rows are reported by their one-based position. Use `--subject history` for single-subject history exports without a subject field.

## Output

| File | Contents |
|---|---|
| `summary.json` | Input, retained, exact duplicate and review counts |
| `retained.json` | Original records including unresolved semantic candidates |
| `exact_duplicates.json` | Removed originals, input row and retained record ID |
| `review_candidates.json` | Candidate pairs, both scores and question/answer text |

Exact matches use normalized subject, question, answer, and ordered options when supplied. The first record is retained. The model embeds question text and correct-answer text separately. Same-subject question similarity retrieves candidates, and answer similarity adds review context. An answer mismatch can signal a conflicting key and remains visible for review. The search spans chapter and topic boundaries.

The defaults of `--threshold 0.70 --top-k 12` are starting values. A record without a candidate can still have an undetected duplicate. Use a new output directory to rerun with different settings.

Phase 1 handles a single batch. Existing MongoDB records can be included through a JSON export. Live MongoDB sync, incremental cross-batch matching, MCP review and final semantic deduplication are future work.

## Tests

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

The integration test uses embedded Qdrant and deterministic test embeddings, so it runs without downloading the model.
