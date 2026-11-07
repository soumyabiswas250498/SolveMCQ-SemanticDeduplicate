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

Phase 1 handles a single batch. Existing MongoDB records can be included through a JSON export. Phase 4 adds opt-in cross-batch matching and manual MongoDB-to-Qdrant reconciliation.

## Tests

```sh
python -m pip install -e '.[review]'
python -m unittest discover -s tests -v
```

The integration test uses embedded Qdrant and deterministic test embeddings, so it runs without downloading the model.

## Phase 2 review via MCP

Run Phase 1 for the input batch, then point the review server at the generated
`review_candidates.json`. Each server instance serves one report.

```sh
python -m pip install -e '.[embeddings,review]'
solvemcq questions.json --subject history --output output/history
solvemcq-review output/history/review_candidates.json --transport streamable-http
```

Test with an MCP client at `http://127.0.0.1:8765/mcp`. For a client that
launches MCP servers as subprocesses, omit `--transport` to use stdio.
For ChatGPT Work, connect the local server through a private
[Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
and add it under Plugins → Add custom MCP server → Tunnel. This server binds to
loopback and relies on the local or private connection for access control.
Keep the endpoint private.

| Tool | Purpose |
|---|---|
| `get_review_summary` | Show pending, duplicate, distinct and deferred pair counts |
| `get_review_batch` | Fetch up to 25 pending pairs or revisit the `needs_review` queue |
| `submit_decisions` | Save verdicts with reasons and a keeper ID for each duplicate |

Example Work prompt:

> Review the next 10 SolveMCQ candidate pairs. Save a duplicate decision
> when both questions test the same fact, distinct when they test different
> facts, and needs_review when a conflicting answer or missing context
> prevents a decision. Tell me how many pairs remain.

Decisions persist in `.solvemcq/reviews.sqlite3` (override with `--db`). Restart
with the same report and DB to resume. A changed report starts a separate review
job. Identical retries are safe, and a conflicting final verdict rejects the
entire submission. A deferred verdict can be resolved later. Decisions apply
to individual pairs. Applying them to produce the final unique JSON is Phase 3.

## Phase 3 export reviewed questions

From the project directory, use the same original input, subject argument, candidate
report and SQLite database used in Phases 1 and 2. Reinstall the editable package
after updating the repository to register the new command.

```sh
python -m pip install -e '.[review]'
solvemcq-finalize sample-input.json output/history/review_candidates.json \
  --subject history --db .solvemcq/reviews.sqlite3 --output output/history-final
```

The exporter reads existing decisions without regenerating embeddings or writing to
SQLite. It follows compatible duplicate decisions to one keeper. It retains all
members of a group when keeper choices are inconsistent, a reviewed pair within
the group is distinct, or a pending or deferred pair touches the group.

- `unique.json` contains the full retained MCQ records in input order.
- `duplicates.json` contains each removed record and the final keeper's input row and ID.
- `unresolved.json` includes full original records, including options, for pending,
  deferred and blocked cases.
- `summary.json` reports counts and has `complete: false` while cases remain unresolved.

Use the unresolved report to inspect the full choices. You can save decisions for
deferred pairs through Phase 2 and export again to a new directory.

## Phase 4 test against a local MongoDB question bank

The bank workflow reads MongoDB and writes a dedicated Qdrant collection.
Use one bank name per MongoDB source. Run sync and batch matching sequentially.
After an import, edit, or archive in the test database, rerun sync before checking
another batch. Sync scans eligible documents and embeds only new or changed
index payloads. Archived, inactive, soft-deleted, and removed records are
pruned from the index. A failed sync can be retried before matching.

Install [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/),
open it once, and wait until Docker is running. From this project folder:

```sh
docker compose up -d
docker compose ps
docker compose exec mongodb mongosh --quiet --eval 'db.adminCommand({ping: 1})'
curl -fsS http://127.0.0.1:6333/readyz
```

Both test services bind to loopback. The Compose volumes preserve their data
across container restarts. If an earlier standalone `solvemcq-mongo-test`
container is running on port 27018, stop it before starting this stack.

Connect MongoDB Compass to `mongodb://127.0.0.1:27018` and import a small
Extended JSON sample into `solvemcq_test.questions`. Preserve the MongoDB
`_id` and `subjectId` values. The current adapter expects the same flat
question and option fields as Phase 1 and filters on `deletedAt` and
`operationalStatus`. Verify those fields against your real schema before
expanding the test. If you already have a separate local MongoDB, point
`SOLVEMCQ_MONGO_URI` at that instance.

```sh
python -m pip install -e '.[embeddings,review,bank]'
solvemcq-bank --database solvemcq_test --bank local-test \
  --qdrant-url http://127.0.0.1:6333 \
  --baseline-report output/baseline/review_candidates.json
solvemcq incoming.json --bank local-test --qdrant-url http://127.0.0.1:6333 \
  --output output/incoming
```

For an incoming single-subject batch without `subjectId`, add
`--subject <actual-subject-id>` to the comparison and export commands.
The incoming batch and the bank must use the same subject identifier. Check
`bank_subjects_without_records` in the comparison summary for mismatches.
For review, point `solvemcq-review` at the incoming report as in Phase 2.
The optional baseline report can be reviewed separately as an audit of
existing records. It is not an incoming import batch.

Then run:

```sh
solvemcq-finalize incoming.json output/incoming/review_candidates.json \
  --output output/incoming-final
```

`import_ready.json` contains retained incoming records after excluding deferred,
pending and blocked cases. Cross-bank duplicates link to the existing MongoDB
ID in `duplicates.json`. Import separately through your normal app workflow
after reviewing the files. The pipeline itself makes no MongoDB writes.

## Browser app

Install once in your existing project environment:

```sh
python -m pip install -e '.[embeddings,review,bank]'
```

On macOS, double-click **Open-SolveMCQ.command** in Finder. You can also start
the app with `solvemcq-ui` and open `http://127.0.0.1:8766` on the same computer.

1. Choose the incoming MCQ JSON. Optionally select a MongoDB Extended JSON
   export as the question bank. The app reads that export directly.
2. Click **Find similar questions**. The first search can take several minutes
   while BGE-M3 downloads and loads.
3. Review the two questions side by side. Choose which to keep if they are the
   same, mark them different, or set the pair aside.
4. Click **Prepare downloads** and download the ready questions. Deferred and
   pending questions, blocked groups, and matching bank IDs stay out of this file.

The browser app saves the current job and decisions under
`.solvemcq/browser`. Use a single running app per data directory. The browser
offers manual review. The existing Codex MCP review flow remains available
through the CLI for AI-assisted review.
