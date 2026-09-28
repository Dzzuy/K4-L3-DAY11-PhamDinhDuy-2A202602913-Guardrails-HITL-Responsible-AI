"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time
from uuid import uuid4


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}
        self._latest_by_user: dict[str, str] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """TODO [DONE]: store input + start timestamp keyed by request_id/user_id."""
        correlation_id = request_id or f"req-{uuid4().hex}"
        self._open[correlation_id] = {
            "request_id": correlation_id,
            "user_id": user_id,
            "input": text,
            "timestamp": utc_now_iso(),
            "started_at": time.perf_counter(),
        }
        self._latest_by_user[user_id] = correlation_id
        return correlation_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """TODO [DONE]: store output, layer decision, latency; append to self.logs."""
        correlation_id = request_id or self._latest_by_user.get(user_id)
        pending = self._open.pop(correlation_id, None) if correlation_id else None
        if pending is None:
            correlation_id = correlation_id or f"req-{uuid4().hex}"
            pending = {
                "request_id": correlation_id,
                "user_id": user_id,
                "input": "",
                "timestamp": utc_now_iso(),
                "started_at": time.perf_counter(),
            }

        if self._latest_by_user.get(user_id) == correlation_id:
            self._latest_by_user.pop(user_id, None)

        latency_ms = max(0.0, (time.perf_counter() - pending["started_at"]) * 1000)
        entry = {
            "request_id": pending["request_id"],
            "timestamp": pending["timestamp"],
            "user_id": pending["user_id"],
            "input": pending["input"],
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": round(latency_ms, 3),
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        # TODO [DONE]: path = filepath or default_audit_log_path()
        #       ensure parent dirs exist, dump self.logs with indent=2
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
