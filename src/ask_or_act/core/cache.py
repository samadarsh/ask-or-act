"""Disk cache of decisions, so the same paid call is never made twice.

Storage is SQLite (Python's standard library): one file, atomic writes, and no server.
The key is a SHA-256 over:

* the decider config: type, model id, prompt version, params
* the example input and options

The example id and the gold label are deliberately **not** in the key. Two examples
with the same text share one answer, and the gold label can't influence what gets
cached. Only successful decisions are cached; failures are retried on the next run.
The cache stores decisions, never input text.
"""

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from ask_or_act.core.types import JSON, DeciderConfig, Decision, PublicExample, sha256_hex

CACHE_SCHEMA = 1


def cache_key(
    config: DeciderConfig, example: PublicExample, extra: dict[str, JSON] | None = None
) -> str:
    """``extra`` distinguishes several calls for one example, e.g. self-consistency
    samples: ``{"sample_index": 3, "temperature": 1.0}``. Without it, k samples of the
    same input would share one key and come back identical."""
    key: dict[str, JSON] = {
        "schema": CACHE_SCHEMA,
        "decider": config.to_json(),
        "input": example.input,
        "options": list(example.options),
    }
    if extra:
        key["extra"] = extra
    return sha256_hex(key)


class DecisionCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS decisions ("
            " key TEXT PRIMARY KEY, decision TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        self._db.commit()

    def get(
        self, config: DeciderConfig, example: PublicExample, extra: dict[str, JSON] | None = None
    ) -> Decision | None:
        row = self._db.execute(
            "SELECT decision FROM decisions WHERE key = ?", (cache_key(config, example, extra),)
        ).fetchone()
        if row is None:
            return None
        stored = Decision.from_json(json.loads(row[0]))
        # Keep the original cost_usd (true cost); this run pays nothing for it.
        return replace(stored, example_id=example.id, cached=True, charged_usd=0.0)

    def put(
        self,
        config: DeciderConfig,
        example: PublicExample,
        decision: Decision,
        extra: dict[str, JSON] | None = None,
    ) -> None:
        if not decision.ok:
            return
        self._db.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?)",
            (
                cache_key(config, example, extra),
                json.dumps(decision.to_json()),
                datetime.now(UTC).isoformat(),
            ),
        )
        self._db.commit()

    def __len__(self) -> int:
        (n,) = self._db.execute("SELECT COUNT(*) FROM decisions").fetchone()
        return int(n)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "DecisionCache":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
