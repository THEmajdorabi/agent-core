"""The run journal: SQLite, one row per step, written before the next call.

This is what makes a run resumable. Every model call, tool call, tool result,
compaction and final answer is appended before the loop moves on, so a crash
loses at most one step.

The subtle part is not writing the journal, it is replaying it. A tool call
that was *started* but whose result was never journalled has unknown outcome.
Re-running it is only safe when the tool said it was idempotent. Otherwise the
honest thing -- and what this module does -- is to hand the model a failed
tool_result explaining that the outcome is unknown, and let it decide. Silently
re-running a non-idempotent side effect is the exact bug this design exists to
prevent.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id  TEXT PRIMARY KEY,
  goal    TEXT NOT NULL,
  status  TEXT NOT NULL,       -- running | complete | failed | interrupted
  started REAL NOT NULL,
  ended   REAL
);

CREATE TABLE IF NOT EXISTS steps (
  run_id        TEXT    NOT NULL,
  step_no       INTEGER NOT NULL,
  ts            REAL    NOT NULL,
  kind          TEXT    NOT NULL,   -- model_call | tool_call_started
                                    -- | tool_result | compaction | final
  payload       TEXT    NOT NULL,   -- JSON
  input_tokens  INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  cost_usd      REAL    DEFAULT 0.0,
  idempotent    INTEGER DEFAULT 0,
  PRIMARY KEY (run_id, step_no)
);

CREATE INDEX IF NOT EXISTS steps_by_run ON steps (run_id, step_no);
"""


@dataclass
class Step:
    """One journalled step, as read back from the database."""

    run_id: str
    step_no: int
    ts: float
    kind: str
    payload: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    idempotent: bool = False


class Journal:
    """Append-only step log for agent runs."""

    def __init__(self, path: str | Path = "runs.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: tools run on a ThreadPoolExecutor, but every
        # write still goes through the loop thread. The connection is not
        # shared for concurrent writes.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL keeps the journal readable while a run is in flight, which is how
        # you inspect a stuck agent without stopping it.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ---- writing ---------------------------------------------------------

    def start_run(self, run_id: str, goal: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, goal, status, started) VALUES (?, ?, ?, ?)",
            (run_id, goal, "running", time.time()),
        )
        self._conn.commit()

    def finish_run(self, run_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE runs SET status = ?, ended = ? WHERE run_id = ?",
            (status, time.time(), run_id),
        )
        self._conn.commit()

    def next_step_no(self, run_id: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(step_no) AS n FROM steps WHERE run_id = ?", (run_id,)
        ).fetchone()
        return 0 if row["n"] is None else int(row["n"]) + 1

    def append(
        self,
        run_id: str,
        kind: str,
        payload: dict[str, Any],
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        idempotent: bool = False,
    ) -> int:
        """Append one step and commit immediately.

        Committing per step is the whole point: a buffered write that is lost
        in a crash is a step the resume logic cannot see.
        """
        step_no = self.next_step_no(run_id)
        self._conn.execute(
            "INSERT INTO steps (run_id, step_no, ts, kind, payload, input_tokens,"
            " output_tokens, cost_usd, idempotent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                step_no,
                time.time(),
                kind,
                json.dumps(payload, ensure_ascii=False),
                input_tokens,
                output_tokens,
                cost_usd,
                int(idempotent),
            ),
        )
        self._conn.commit()
        return step_no

    # ---- reading ---------------------------------------------------------

    def steps(self, run_id: str) -> list[Step]:
        rows = self._conn.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY step_no", (run_id,)
        ).fetchall()
        return [
            Step(
                run_id=row["run_id"],
                step_no=row["step_no"],
                ts=row["ts"],
                kind=row["kind"],
                payload=json.loads(row["payload"]),
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                cost_usd=row["cost_usd"],
                idempotent=bool(row["idempotent"]),
            )
            for row in rows
        ]

    def run_row(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def unfinished_runs(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT run_id FROM runs WHERE status = 'running' ORDER BY started"
        ).fetchall()
        return [row["run_id"] for row in rows]

    def totals(self, run_id: str) -> dict[str, float]:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(input_tokens), 0) AS i, COALESCE(SUM(output_tokens), 0) AS o,"
            " COALESCE(SUM(cost_usd), 0.0) AS c FROM steps WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return {"input_tokens": row["i"], "output_tokens": row["o"], "cost_usd": row["c"]}

    # ---- replay ----------------------------------------------------------

    def completed_tool_call_ids(self, run_id: str) -> set[str]:
        """Call ids that have a journalled result, and so must never re-run."""
        return {
            step.payload["tool_use_id"] for step in self.steps(run_id) if step.kind == "tool_result"
        }

    def orphaned_tool_calls(self, run_id: str) -> list[Step]:
        """Calls journalled as started whose result never landed.

        These are the calls whose outcome is genuinely unknown: the process
        died somewhere between "about to run this" and "here is what it
        returned". `Agent.resume` applies the idempotency rule to each one.
        """
        started = [s for s in self.steps(run_id) if s.kind == "tool_call_started"]
        completed = self.completed_tool_call_ids(run_id)
        return [s for s in started if s.payload["tool_use_id"] not in completed]

    def close(self) -> None:
        self._conn.close()
