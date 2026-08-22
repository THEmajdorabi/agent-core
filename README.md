# agent-core

A tool-calling agent runtime written from scratch — no LangChain, no CrewAI, no LlamaIndex.

For engineers who need to know exactly what happens between a user's question and a model's final answer, because they have to debug it, budget it, or explain it.

The unusual part: **a resumed run refuses to re-run a side effect whose outcome it cannot establish.** Most agent loops replay their journal and re-execute whatever was in flight. This one journals a tool call *before* running it along with the tool's own idempotency declaration, and on resume re-runs only what is safe to re-run. Everything else comes back to the model as "this started, the outcome is unknown, verify before relying on it."

---

## Numbers

Measured on 2026-08-22 with `claude-sonnet-5`, 10 tasks x 3 repeats.
Reproduce with `make benchmark`; raw data in `benchmark/results.json`.

| Metric | Target | Measured |
|---|---|---|
| Tasks solved on the built-in benchmark | >= 7 of 10 | **10 of 10** |
| Attempt solve rate | reported | **100%** (30/30) |
| Median steps per solved task | reported | **5** |
| Median cost per solved task | reported | **$0.0308** |
| Median wall time per solved task | reported | **12.4s** |
| Total cost of the full benchmark | reported | **$1.06** |
| Runs recovered after a forced kill | 10 of 10 | **10 of 10** |
| Test suite passes with zero paid API calls | required | **94 passed, 10 deselected** |
| Sandbox isolation properties verified | all | **10 of 10** |

### Per task

| Task | Solved | Median steps | Median cost | Median wall |
|---|---|---|---|---|
| `01_off_by_one_slice` | 3/3 | 8 | $0.0523 | 30.8s |
| `02_wrong_default_argument` | 3/3 | 4 | $0.0216 | 9.2s |
| `03_missing_edge_case` | 3/3 | 5 | $0.0255 | 9.7s |
| `04_import_cycle` | 3/3 | 6 | $0.0372 | 14.2s |
| `05_swallowed_exception` | 3/3 | 4 | $0.0261 | 11.0s |
| `06_implement_from_docstring` | 3/3 | 5 | $0.0492 | 23.5s |
| `07_regex_fails_on_arabic` | 3/3 | 6 | $0.0386 | 19.8s |
| `08_timezone_bug` | 3/3 | 5 | $0.0294 | 11.0s |
| `09_schema_mismatch` | 3/3 | 6 | $0.0387 | 17.0s |
| `10_mutable_default` | 3/3 | 4 | $0.0218 | 8.8s |

Every task was solved on every one of three attempts. The verdict is `check.sh`
exiting 0, never the agent's own claim that it fixed something.

### What *is* measured

| Check | Script | Result |
|---|---|---|
| All 10 benchmark tasks fail unfixed and pass with a reference fix | [`scripts/verify_benchmark_tasks.py`](scripts/verify_benchmark_tasks.py) | 10 / 10 |
| Crash-and-resume never repeats a side effect | [`tests/test_crash_resume.py`](tests/test_crash_resume.py) | 10 / 10 kill points |
| Compaction never splits a `tool_use` from its `tool_result` | [`tests/test_context.py`](tests/test_context.py) | pass |
| Full test suite, no paid API calls | `make test` | 94 passed, 10 deselected |
| Sandbox isolation, incl. an enforced 512 MiB cap | [`tests/test_sandbox.py`](tests/test_sandbox.py) | 10 / 10 (`make test-integration`) |

The crash test spawns a real subprocess and kills it uncatchably at ten
different delays. **Six of the ten land mid-side-effect** — the dangerous case
where the write already happened but its result was never journalled. All ten
resume without repeating a side effect, and the four that orphan an
*idempotent* call do re-run it, so the test cannot be passed by a resume that
simply refuses to do anything.

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

A completed run renders as a tree. This is **real output** from
`python scripts/render_trace.py --demo`, which runs the actual loop with the
actual filesystem and shell tools against a scripted model, so it needs no API
key:

```
run eaee76bb  goal='fix the failing test'  4 steps  4 tool calls  $0.0393  0.1s
├─ model_call         3,000 in / 189 out
├─ tool: list_dir     0.00s  ok
├─ tool: read_file    0.00s  ok
├─ model_call         2,100 in / 402 out
├─ tool: run_command  0.05s  ok
├─ model_call         2,100 in / 288 out
├─ tool: write_file   0.00s  ok
├─ model_call         1,200 in / 64 out
└─ final              'Fixed: the range excluded the final window. Changed `len(...'
```

The token counts and the cost come from the scripted completions and a real
price table, so the dollar figure is arithmetic on made-up token counts, not a
measured spend. `steps` counts model calls, matching `RunResult.steps` and what
the guardrails budget against.

A journalled run can be re-rendered later, after the process that ran it has
exited:

```bash
uv run agent-core trace <run_id>
```

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
properties **from inside the container** rather than trusting that the flags
did anything. All ten pass on this host:

| Property | Flag | What the test observes |
|---|---|---|
| No network | `--network none` | connecting to `1.1.1.1:53` raises `OSError` |
| Read-only root | `--read-only` | appending to `/etc/passwd` raises `OSError` |
| Writable scratch | `--tmpfs /workspace` | write-then-read round-trips |
| Non-root | `--user 65534:65534` | `os.getuid()` → `65534` |
| Process cap | `--pids-limit 128` | `/sys/fs/cgroup/pids.max` → `128` |
| **Memory cap** | `--memory 512m` | `/sys/fs/cgroup/memory.max` → **536870912** = exactly 512 MiB; a 2 GB allocation is **OOM-killed, exit 137** |
| Timeout kill | wall-clock | `sleep(60)` under a 3s limit → `timed_out=True`, exit 124 |
| Output truncation | 10 KB cap | 5,000 chars under a 200-byte cap returns a truncation marker |

The memory row is the one worth checking rather than asserting: **Docker
silently ignores `--memory` on hosts without the right cgroup support.** This
host runs cgroups v2, the cgroup file reports the exact byte count, and the
kernel kills a process that exceeds it. `verify_memory_limit()` reproduces it
and `make setup` runs it. Had it come back unenforced, the right fix would have
been to delete the claim, not the test.

One sandbox test needs no daemon and *does* run in the default suite: the check
that `run_python` raises rather than silently falling back to host Python. That
is the property whose failure would be dangerous rather than merely unproven,
so it must not hide behind infrastructure that might be absent.

```bash
make test               # 94 passed, 10 deselected
make test-integration   # 10 passed  (needs a Docker daemon)
pytest -q -m ""         # 104 passed (everything)
```

Getting here needed three fixes on the build machine — a missing WSL 2 kernel,
a bad Docker Desktop install, and, the actual blocker, **hardware
virtualization disabled in firmware**. That last one is in
[`DECISIONS.md`](DECISIONS.md) along with the diagnostic mistake that made it
take longer than it should have.

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
