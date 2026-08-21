# Every target works from a clean clone. `uv` is the only prerequisite.
.PHONY: setup test test-integration test-all lint format typecheck run eval benchmark sandbox-image verify-tasks clean

PY := uv run

setup:  ## create the venv, install deps, build the sandbox image
	uv venv --python 3.12
	uv pip install -e ".[dev]"
	@echo "building the sandbox image (skipped if docker is unavailable)"
	-$(PY) python -c "from agent_core.tools import docker_available; \
	  from agent_core.tools.python_sandbox import build_sandbox_image; \
	  from agent_core.config import load_settings; \
	  s = load_settings(); \
	  build_sandbox_image('docker/sandbox.Dockerfile', s.sandbox.image) if docker_available() \
	  else print('docker unavailable: run_python will not be registered')"
	@echo "setup complete. Copy .env.example to .env and add your API keys."

test:  ## run the test suite (no paid API calls)
	$(PY) pytest -q

test-integration:  ## sandbox + live-API tests; needs Docker and/or an API key
	$(PY) pytest -q -m integration

test-all:  ## everything, including integration
	$(PY) pytest -q -m ""

lint:
	$(PY) ruff check .
	$(PY) ruff format --check .

format:
	$(PY) ruff format .
	$(PY) ruff check --fix .

typecheck:
	$(PY) mypy src/agent_core

run:  ## GOAL="..." make run
	$(PY) agent-core run "$(GOAL)"

verify-tasks:  ## prove every benchmark task is broken and solvable
	$(PY) python scripts/verify_benchmark_tasks.py

benchmark:  ## the README numbers; needs ANTHROPIC_API_KEY
	$(PY) python scripts/run_benchmark.py --repeats 3

eval: benchmark  ## alias required by the spec

benchmark-dry:  ## exercise the harness with no API calls
	$(PY) python scripts/run_benchmark.py --dry-run --repeats 1

sandbox-image:
	docker build -f docker/sandbox.Dockerfile -t agent-core-sandbox:latest docker/

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache runs.db benchmark/results.dry-run.json
