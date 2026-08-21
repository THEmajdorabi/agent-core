"""Typed settings loaded from config.yaml, with secrets from the environment.

Two rules this module exists to enforce:
  1. No magic numbers scattered through the code. They live in config.yaml.
  2. No secrets in config.yaml. API keys come from the environment only.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class ModelSpec(BaseModel):
    provider: str
    id: str
    max_tokens: int = 4096


class Price(BaseModel):
    """USD per 1,000,000 tokens."""

    input: float
    output: float


class GuardrailSettings(BaseModel):
    max_steps: int = 25
    max_cost_usd: float = 1.00
    max_wall_s: float = 300.0
    max_consecutive_tool_failures: int = 3


class ContextSettings(BaseModel):
    max_tokens: int = 120_000
    trigger_ratio: float = 0.8
    target_ratio: float = 0.6
    keep_recent: int = 6


class SandboxSettings(BaseModel):
    image: str = "agent-core-sandbox:latest"
    memory: str = "512m"
    cpus: str = "1"
    pids_limit: int = 128
    timeout_s: float = 15.0
    max_output_bytes: int = 10240


class JournalSettings(BaseModel):
    path: str = "runs.db"


class Settings(BaseModel):
    models: dict[str, ModelSpec]
    pricing: dict[str, Price]
    guardrails: GuardrailSettings = Field(default_factory=GuardrailSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    journal: JournalSettings = Field(default_factory=JournalSettings)

    def price_for(self, model_id: str) -> Price:
        """Price of a model, or zero if we have no entry.

        Returning zero rather than raising is deliberate: an unknown model must
        not crash a run. It does mean an unpriced model reports $0.00, so
        `cost_is_estimated` on RunResult flags that case for the caller.
        """
        return self.pricing.get(model_id, Price(input=0.0, output=0.0))

    def is_priced(self, model_id: str) -> bool:
        return model_id in self.pricing


def config_path() -> Path:
    """Locate config.yaml: explicit override, else the repo root."""
    override = os.environ.get("AGENT_CORE_CONFIG")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "config.yaml"


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Read and validate config.yaml. Cached: config does not change mid-run."""
    path = config_path()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return Settings.model_validate(raw)


def require_env(name: str) -> str:
    """Fetch a required secret from the environment with an actionable error."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in, "
            f"or export {name} in your shell. Keys are never read from config.yaml."
        )
    return value
