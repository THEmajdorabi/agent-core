# agent-core

A tool-calling agent runtime written from scratch — no LangChain, no CrewAI, no LlamaIndex.

For engineers who need to know exactly what happens between a user's question and a model's final answer, because they have to debug it, budget it, or explain it.

The unusual part: **a resumed run refuses to re-run a side effect whose outcome it cannot establish.** Most agent loops replay their journal and re-execute whatever was in flight. This one journals a tool call *before* running it along with the tool's own idempotency declaration, and on resume re-runs only what is safe to re-run. Everything else comes back to the model as "this started, the outcome is unknown, verify before relying on it."

---

## Numbers

> **Status: not yet measured.** The benchmark harness is complete and runs end
> to end, but no `ANTHROPIC_API_KEY` was available in the build environment, so
> no live run has happened. Every figure below is `TBD` rather than a guess.
> Reproduce with `make benchmark` once a key is set.

| Metric | Target | Measured |
|---|---|---|
| Tasks solved on the built-in benchmark | ≥ 7 of 10 | TBD |
| Median steps per solved task | reported | TBD |
| Median cost per solved task (USD) | reported | TBD |
| Runs recovered after a forced kill | 10 of 10 | **8 of 8 kill points** (see below) |
| Test suite passes with zero paid API calls | required | **88 passed, 10 skipped** |

Produced by [`scripts/run_benchmark.py`](scripts/run_benchmark.py) → `benchmark/results.json`.

### What *is* measured

| Check | Script | Result |
|---|---|---|
| All 10 benchmark tasks fail unfixed and pass with a reference fix | [`scripts/verify_benchmark_tasks.py`](scripts/verify_benchmark_tasks.py) | 10 / 10 |
| Crash-and-resume never repeats a side effect | [`tests/test_crash_resume.py`](tests/test_crash_resume.py) | 4 kill points × 2 properties, all pass |
| Compaction never splits a `tool_use` from its `tool_result` | [`tests/test_context.py`](tests/test_context.py) | pass |
| Full test suite, no paid API calls | `make test` | 88 passed, 10 skipped (Docker) |

The crash test spawns a real subprocess and kills it uncatchably at four
different delays. Two of those land mid-side-effect — the dangerous case where
the write already happened but its result was never journalled.

---

## What it does

```
User goal
   │
   ▼
┌─────────────────────────────────────────────────┐
│  Agent.run(goal)                                │
│    loop:                                        │
│      1. call model with messages + tool schemas │
│      2. if stop_reason != "tool_use": finish    │
│      3. execute requested tools (in parallel)   │
│      4. append tool_result blocks               │
│      5. compact context if over budget          │
│      6. check guardrails (steps, cost, time)    │
└─────────────────────────────────────────────────┘
   │            │               │
   ▼            ▼               ▼
ToolRegistry  Journal(SQLite)  Tracer(spans)
```

A completed run renders as a tree:

```
run 3f2a1b8c  goal='fix the failing test'  6 steps  $0.0312  47.3s
├─ model_call        1,204 in / 189 out
├─ tool: list_dir    0.02s  ok
├─ tool: read_file   0.01s  ok
├─ model_call        2,890 in / 402 out
├─ tool: run_command 1.31s  ERROR AssertionError
├─ model_call        3,110 in / 288 out
├─ tool: write_file  0.01s  ok
└─ final             'Fixed: off-by-one in the slice bound'
```

*(Shape shown from `Tracer.render`; the figures above are from a scripted test run, not a live benchmark. A live screenshot goes here once a key is available.)*

---

## Quick start

```bash
git clone <this repo> && cd agent-core
make setup
cp .env.example .env   # add ANTHROPIC_API_KEY
make test
```

Run an agent against a directory:

```bash
uv run agent-core run "fix the failing test in tests/" --workdir ./my-project --allow-shell
```

Resume an interrupted run:

```bash
uv run agent-core runs
uv run agent-core resume <run_id>
```

Re-render a past run's trace:

```bash
uv run agent-core trace <run_id>
```

Without `make`, every target is a one-liner — see the [Makefile](Makefile). For example:

```bash
uv run pytest -q
```

---

## The parts worth reading

### `registry.py` — schemas you never write

A tool is a plain function. The decorator builds a pydantic model from its
signature and calls `.model_json_schema()`, so unions, `Literal`, `Enum`,
nested models and defaults are handled by a library that already gets them
right.

```python
@registry.register(timeout_s=10)
def search_corpus(query: str, top_k: int = 5) -> list[dict]:
    """Search the document corpus and return matching passages."""
```

Registration fails loudly if the docstring is missing, because the docstring
*is* the description the model uses to decide when to call the tool.
`Annotated[str, "..."]` documents individual parameters.

### `journal.py` + `agent.py` — resume without repeating yourself

Each tool call is journalled as `tool_call_started` — carrying the tool's
`idempotent` flag — *before* it executes. On resume:

| State in the journal | What resume does |
|---|---|
| started **and** result recorded | nothing; replayed from the journal |
| started, no result, `idempotent=True` | re-executed |
| started, no result, `idempotent=False` | **not** re-run; the model gets an error result saying the outcome is unknown |

That third row is the whole point. Silently re-running a non-idempotent side
effect is the bug this design exists to prevent.

### `context.py` — compaction that cannot corrupt a transcript

Compaction summarises old turns when the transcript passes a fraction of the
window. The hard part is *where* it is allowed to cut. A `tool_use` block
separated from its `tool_result` produces an opaque API error several steps
later.

So `settled_boundaries()` returns only the indices where every tool call
raised so far has been answered, and the compactor cuts at one of those or
declines to cut at all. Pinned facts — recorded by the agent through a
`pin_fact` tool — are re-injected verbatim after every pass.

Token counting uses the provider's own counter (Anthropic's `count_tokens`
endpoint, `tiktoken` for OpenAI). The fallback heuristic counts Arabic and
Latin characters at **different** chars-per-token ratios, because a ratio
calibrated on English under-estimates an Arabic transcript badly enough to
blow the window before compaction ever fires.

### `tools/python_sandbox.py` — isolation that is checked, not claimed

Model-written Python runs in Docker with `--network none`, `--memory 512m`,
`--cpus 1`, `--read-only`, `--pids-limit 128` and a non-root user.

[`tests/test_sandbox.py`](tests/test_sandbox.py) verifies each of those
properties from inside the container rather than trusting the flags — including
`verify_memory_limit()`, because Docker silently ignores `--memory` on some
hosts and a README claim you cannot demonstrate is worse than no claim.

**These eleven tests have not run on the build machine.** Docker Desktop there
could not start (a missing WSL 2 kernel, since installed, needing a reboot),
so they skip. They run for real in CI on `ubuntu-latest`, which builds the
image and executes them. Treat the isolation properties as CI-verified, not
locally verified.

There is **no host-Python fallback**. If Docker is missing, `run_python` is not
registered and calling it raises. A sandbox that quietly stops sandboxing is
worse than one that is plainly unavailable.

---

## The benchmark

Ten small broken repositories in [`benchmark/tasks/`](benchmark/tasks), each
with a `check.sh` that exits 0 when fixed:

| # | Task | Bug |
|---|---|---|
| 01 | `off_by_one_slice` | slice bound excludes the last window |
| 02 | `wrong_default_argument` | money rounds to 0 places instead of 2 |
| 03 | `missing_edge_case` | `mean([])` divides by zero |
| 04 | `import_cycle` | two modules import each other at module level |
| 05 | `swallowed_exception` | `except Exception` hides malformed JSON |
| 06 | `implement_from_docstring` | sentence-aware chunker, unimplemented |
| 07 | `regex_fails_on_arabic` | `[A-Za-z]+` finds no Arabic words |
| 08 | `timezone_bug` | naive `utcnow()` compared to an aware datetime |
| 09 | `schema_mismatch` | producer and consumer disagree on field names |
| 10 | `mutable_default` | `basket=[]` shared across calls |

The verdict comes from `check.sh`, never from the agent's own claim that it
fixed something.

`scripts/verify_benchmark_tasks.py` closes the other end, proving each task
**fails** unfixed and **passes** with a reference fix — on a copy, so the
originals stay broken. A task that already passes measures nothing.

```bash
make verify-tasks     # 10/10 broken-then-solvable
make benchmark-dry    # exercise the harness with no API calls
make benchmark        # the real numbers; needs a key
```

---

## Configuration

Model IDs and prices live in [`config.yaml`](config.yaml) with the date they
were checked against provider docs (2026-08-22). Never hardcoded in source, so
a price change is a one-line edit rather than a hunt.

Secrets come from the environment only, via `python-dotenv`. `.env` has been
gitignored since the first commit.

---

## Known limits

Stated here rather than discovered by a reviewer:

- **No live benchmark numbers yet.** No API key in the build environment. The
  harness runs end to end in `--dry-run`.
- **A timed-out tool leaks its thread.** Python cannot interrupt arbitrary
  running code. The loop is unblocked immediately and the model gets its
  timeout result, but the worker runs to completion in the background.
  `run_python` does not have this problem — Docker can kill a process.
- **Compaction may decline to compact.** When no settled boundary exists inside
  the allowed range, the transcript is left alone. Correct, but it means
  compaction is not a guarantee against overflow.
- **The token heuristic is uncalibrated.** The Arabic and Latin ratios are
  reasoned starting values, not measured ones. Calibrating them needs an API
  key.
- **`docker-compose.yml` mounts the Docker socket**, which grants the container
  root-equivalent access to the host. Fine for a local demo, not for production.

## Decisions

[`DECISIONS.md`](DECISIONS.md) records each significant choice, the alternative
rejected, and why — including the ones that were rejected after being tried.
