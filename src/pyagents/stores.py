"""Run stores: persistence and checkpoint/resume support.

A *run store* records finished (or crashed) runs so they can be inspected,
replayed or resumed later. Two implementations ship with the package:

* :class:`MemoryStore` - in-process dictionary, ideal for tests and notebooks.
* :class:`SQLiteRunStore` - durable, zero-dependency, writes JSON blobs to a
  local database file.

Both implement the :class:`RunStore` protocol, which is all
:class:`~pyagents.agent.Agent` needs to checkpoint itself:

    >>> from pyagents.stores import MemoryStore
    >>> store = MemoryStore()
    >>> store.list_runs()
    []
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .errors import RunNotFound
from .types import RunResult, utc_now

__all__ = ["MemoryStore", "RunStore", "RunStoreLike", "SQLiteRunStore", "store_from_uri"]


class RunStore(ABC):
    """Persistence interface for :class:`~pyagents.types.RunResult` objects."""

    @abstractmethod
    def save(self, result: RunResult, *, event: str = "checkpoint") -> None:
        """Persist (or overwrite) a run."""

    @abstractmethod
    def load(self, run_id: str) -> RunResult:
        """Load a run by id.

        Raises:
            RunNotFound: when the run is unknown.
        """

    @abstractmethod
    def list_runs(self, *, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return lightweight summaries of stored runs, newest first."""

    @abstractmethod
    def delete(self, run_id: str) -> None:
        """Remove a run; missing ids are ignored."""

    def has(self, run_id: str) -> bool:
        """Return ``True`` when a run with this id is stored."""
        try:
            self.load(run_id)
            return True
        except RunNotFound:
            return False

    def close(self) -> None:  # noqa: B027 - optional hook with a no-op default
        """Release any resources (no-op for in-memory stores)."""

    def __bool__(self) -> bool:
        # A store is always truthy, even when it holds no runs yet - otherwise
        # `if store:` would silently skip persistence for empty stores.
        return True


#: Anything that behaves like a run store.
RunStoreLike = Union[RunStore, Any]


class MemoryStore(RunStore):
    """In-process run store backed by a dictionary."""

    def __init__(self, results: Optional[Iterable[RunResult]] = None) -> None:
        self._runs: Dict[str, RunResult] = {}
        for result in results or []:
            self.save(result)

    def save(self, result: RunResult, *, event: str = "checkpoint") -> None:
        """Store a deep copy of ``result`` under its run id."""
        clone = RunResult.from_dict(result.to_dict())
        clone.metadata.setdefault("checkpoints", []).append({"event": event, "at": utc_now()})
        self._runs[result.run_id] = clone

    def load(self, run_id: str) -> RunResult:
        """Return a copy of the stored run."""
        if run_id not in self._runs:
            raise RunNotFound(f"No stored run with id '{run_id}'", run_id=run_id)
        return RunResult.from_dict(self._runs[run_id].to_dict())

    def list_runs(self, *, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List stored runs, newest first."""
        items = sorted(self._runs.values(), key=lambda item: item.started_at, reverse=True)
        if status:
            items = [item for item in items if item.status.value == _normalise_status(status)]
        return [
            {
                "run_id": item.run_id,
                "status": item.status.value,
                "agent": item.agent,
                "output": item.output[:200],
                "steps": len(item.steps),
                "started_at": item.started_at,
                "total_tokens": item.usage.total_tokens,
            }
            for item in items[:limit]
        ]

    def delete(self, run_id: str) -> None:
        """Delete a stored run."""
        self._runs.pop(run_id, None)

    def __len__(self) -> int:
        return len(self._runs)


class SQLiteRunStore(RunStore):
    """Durable run store backed by SQLite.

    Args:
        path: Database file. Use ``":memory:"`` for a throwaway database.
        check_same_thread: Leave this alone unless you have a reason to change
            it - the store serialises access with its own lock, so one
            connection can safely serve several threads (which happens
            whenever ``Agent.run`` executes its async loop on a worker thread).
    """

    def __init__(
        self, path: Union[str, Path] = "pyagents_runs.sqlite3", *, check_same_thread: bool = False
    ) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=check_same_thread)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                agent TEXT,
                output TEXT,
                started_at REAL,
                finished_at REAL,
                steps INTEGER,
                total_tokens INTEGER,
                payload TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._connection.commit()

    def save(self, result: RunResult, *, event: str = "checkpoint") -> None:
        """Insert or update a run row (thread-safe)."""
        payload = json.dumps(result.to_dict(), ensure_ascii=False)
        with self._lock:
            self._execute(
                """
                INSERT INTO runs (run_id, status, agent, output, started_at, finished_at, steps,
                                  total_tokens, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    output=excluded.output,
                    finished_at=excluded.finished_at,
                    steps=excluded.steps,
                    total_tokens=excluded.total_tokens,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    result.run_id,
                    result.status.value,
                    result.agent,
                    result.output,
                    result.started_at,
                    result.finished_at,
                    len(result.steps),
                    result.usage.total_tokens,
                    payload,
                    utc_now(),
                ),
            )

    def load(self, run_id: str) -> RunResult:
        """Load a run from the database."""
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RunNotFound(f"No stored run with id '{run_id}'", run_id=run_id)
        return RunResult.from_dict(json.loads(row[0]))

    def list_runs(self, *, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List stored runs, newest first."""
        query = "SELECT run_id, status, agent, output, started_at, steps, total_tokens FROM runs"
        params: List[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [
            {
                "run_id": row[0],
                "status": row[1],
                "agent": row[2],
                "output": (row[3] or "")[:200],
                "started_at": row[4],
                "steps": row[5],
                "total_tokens": row[6],
            }
            for row in rows
        ]

    def delete(self, run_id: str) -> None:
        """Delete a run row."""
        with self._lock:
            self._execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._connection.close()

    def _execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        """Run a write statement and commit."""
        self._connection.execute(sql, params)
        self._connection.commit()

    def __enter__(self) -> SQLiteRunStore:  # pragma: no cover - convenience
        return self

    def __exit__(self, *exc_info: Any) -> None:  # pragma: no cover - convenience
        self.close()


def _normalise_status(status: Optional[str]) -> str:
    """Normalise a status filter such as ``"RunStatus.FAILED"`` to ``"failed"``."""
    return str(status or "").split(".")[-1].strip().lower()


def store_from_uri(uri: Union[str, Path, None]) -> Optional[RunStore]:
    """Build a store from a shorthand URI.

    ``None`` disables persistence; ``"memory:"`` returns a
    :class:`MemoryStore`; anything else is treated as a SQLite file path.

    >>> isinstance(store_from_uri("memory:"), MemoryStore)
    True
    """
    if uri is None:
        return None
    text = str(uri)
    if text in {"memory:", "memory", ":memory:"}:
        return MemoryStore()
    return SQLiteRunStore(text)
