# Conventions

The rules this repository is written to. Deviations forced by the build
environment are recorded in DECISIONS.md.

## 1. Language, versions, tooling

- Python **3.12**.
- Dependency management: **uv**. `uv init`, `uv add <pkg>`, `uv run <cmd>`.
  Fall back to `pip` + `requirements.txt` only if uv is unavailable.
- Formatting and linting: **ruff** (`ruff format`, `ruff check --fix`).
- Type checking: **mypy** in non-strict mode. Public functions carry type hints.
- Tests: **pytest**. Test files mirror source paths under `tests/`.
- Every repo has a `Makefile` with at least: `setup`, `test`, `lint`, `run`, `eval`.

## 2. Repo layout

```
<repo>/
  README.md
  CONVENTIONS.md
  DECISIONS.md
  Makefile
  pyproject.toml
  .env.example          # every variable, no values
  .gitignore            # .env, __pycache__, .venv, data/raw, models/, *.db
  docker-compose.yml
  Dockerfile
  src/<package>/        # all source, importable package
  tests/
  scripts/              # one-off runnable scripts that produce README numbers
  data/                 # gitignored except data/README.md explaining how to get it
  docs/
```

## 3. Configuration

- All configuration lives in `config.yaml`, loaded into a **pydantic** `Settings`
  model at startup. No magic numbers scattered through the code.
- Secrets come from environment variables only. `python-dotenv` loads `.env` in
  development. `.env` is never committed.
- **Model IDs live in `config.yaml`, never hardcoded in source.** Provider model
  names change. Look up the current IDs in the Anthropic and OpenAI docs when you
  first write the config, and note the date you checked in a comment.

Example:

```yaml
# checked against provider docs on <DATE>
models:
  answer:  { provider: anthropic, id: "<current-sonnet-id>", max_tokens: 2048 }
  judge:   { provider: anthropic, id: "<current-sonnet-id>", max_tokens: 1024 }
  compare: { provider: openai,    id: "<current-gpt-id>",    max_tokens: 2048 }
```

## 4. Hardware profiles (required in repos 02 and 04)

Hardware is not known ahead of time. At startup, detect the device and select a
profile. Log which profile was chosen. Never crash because a GPU is absent.

```python
def detect_profile() -> str:
    import torch

    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        return "cuda_large" if vram_gb >= 12 else "cuda_small"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

| Profile | Embedding model | Dense dims | Reranker | Rerank candidates |
|---|---|---|---|---|
| `cuda_large` | `BAAI/bge-m3` (fp16) | 1024 | `BAAI/bge-reranker-v2-m3` | top 50 |
| `cuda_small` | `BAAI/bge-m3` (fp16) | 1024 | `BAAI/bge-reranker-v2-m3` | top 30 |
| `mps` | `BAAI/bge-m3` | 1024 | `BAAI/bge-reranker-v2-m3` | top 20, batch 8 |
| `cpu` | `intfloat/multilingual-e5-base` | 768 | `BAAI/bge-reranker-v2-m3` (int8 ONNX) | top 20 |

**The reranker never changes across profiles.** `BAAI/bge-reranker-base` is
English and Chinese only; on Arabic it reranks close to randomly, which would
silently poison the ablation table. Weaker hardware shrinks the *candidate set*,
never swaps in a monolingual model. If `bge-reranker-v2-m3` is still too slow on
CPU, the correct fallback is disabling reranking entirely and reporting that row
honestly, or trying `jinaai/jina-reranker-v2-base-multilingual`.

**Embedding dimensions differ across profiles (1024 vs 768), so the vector
collection name must encode the model**: `chunks__bge_m3_1024` versus
`chunks__e5_base_768`. Store the model id and dimension in the collection config
and assert them at startup, so a profile switch fails loudly instead of at query
time. Note also that `multilingual-e5-base` has a **512-token limit** against
BGE-M3's 8192, so on the CPU profile the heading-path prefix plus chunk body must
stay under 512 tokens.

Rules:

- **The CPU profile must work.** It is the fallback that keeps the demo runnable
  anywhere. Slower is fine; broken is not.
- Fine-tuning in repo 04 detects no GPU and prints clear instructions for running
  `scripts/train.py` on Colab or Kaggle instead. It does not silently attempt a
  multi-hour CPU training run.
- Record measured throughput per profile in the README. Different numbers per
  profile is a strength, not a mess.

## 5. Code style

- Small modules with one clear responsibility. If a file passes ~300 lines, split it.
- Public functions get docstrings stating what, not how.
- No bare `except:`. Catch specific exceptions and log with context.
- All I/O boundaries (API calls, disk, network) get explicit timeouts and retries
  with exponential backoff. Use `tenacity`.
- Structured logging with `structlog`. Log events, not sentences.
- No `print()` outside `scripts/`.

## 6. Testing

- Every module gets tests. Target meaningful coverage of logic, not a percentage.
- **Never call a paid API in a unit test.** Record fixtures once and replay them.
- Integration tests that do hit an API are marked `@pytest.mark.integration` and
  excluded from the default `make test` run.
- Arabic text handling gets its own test file with real Arabic strings covering
  diacritics, alef variants, taa marbuta, tatweel and Arabic-Indic digits.

## 7. Git discipline

- Small commits, imperative present tense: `add reciprocal rank fusion`.
- Commit after each working increment, not at the end of the day.
- Branch per feature, PR into main, CI must be green to merge.
- The commit history is part of the portfolio. It should look like someone worked.

## 8. Documentation as you go

- `DECISIONS.md` gets an entry every time you choose between real alternatives:

  ```
  ## 2026-09-03 BM25 stemmer
  Chose: light stemming (strip common affixes only)
  Rejected: ISRI root stemmer
  Why: ISRI over-stems legal terminology; recall@10 dropped 0.71 -> 0.63 on the
  eval set. Measured with scripts/ablate_retrieval.py.
  ```

- A failed approach that you removed still gets an entry. Those entries are what
  make the repo read as engineering rather than assembly.

## 9. Things that are never acceptable

- A number in a README that no script in the repo can reproduce.
- A committed API key, even a revoked one.
- A `try/except: pass`.
- A model ID hardcoded in three different files.
- A README that describes features that do not exist yet.

