"""Focused regression tests for the mandatory Checkpoint 3 contracts."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from google.genai import types

from assignment.audit_log import AuditLogPlugin
from assignment.monitoring import MonitoringAlert
from assignment.pipeline import build_production_plugins, is_egress_allowed
from assignment.rate_limiter import RateLimitPlugin


class _Context:
    def __init__(self, user_id: str):
        self.user_id = user_id


def _message(text: str = "What is my account balance?") -> types.Content:
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def test_rate_limit_is_per_user_and_expires_old_requests():
    async def exercise():
        limiter = RateLimitPlugin(max_requests=2, window_seconds=10)
        with patch(
            "assignment.rate_limiter.time.time",
            side_effect=[0.0, 1.0, 2.0, 2.1, 11.1],
        ):
            assert await limiter.on_user_message_callback(
                invocation_context=_Context("alice"), user_message=_message()
            ) is None
            assert await limiter.on_user_message_callback(
                invocation_context=_Context("alice"), user_message=_message()
            ) is None
            assert await limiter.on_user_message_callback(
                invocation_context=_Context("alice"), user_message=_message()
            ) is not None
            assert await limiter.on_user_message_callback(
                invocation_context=_Context("bob"), user_message=_message()
            ) is None
            assert await limiter.on_user_message_callback(
                invocation_context=_Context("alice"), user_message=_message()
            ) is None

    asyncio.run(exercise())


def test_audit_and_monitoring_exports(tmp_path):
    audit = AuditLogPlugin()
    request_id = audit.record_input(user_id="student", text="account balance")
    row = audit.record_output(
        user_id="student",
        text="allowed",
        request_id=request_id,
    )
    assert row["request_id"] == request_id
    assert row["input"] == "account balance"
    assert row["latency_ms"] >= 0

    monitor = MonitoringAlert()
    monitor.total_requests = 10
    monitor.blocked_requests = 6
    monitor.rate_limit_hits = 5
    assert len(monitor.check_metrics()) == 2
    assert len(monitor.check_metrics()) == 2  # no duplicate alerts

    audit_path = audit.export_json(str(tmp_path / "audit.json"))
    metrics_path = monitor.export_json(str(tmp_path / "metrics.json"))
    assert json.loads(audit_path.read_text())[0]["request_id"] == request_id
    assert json.loads(metrics_path.read_text())["block_rate"] == 0.6


def test_plugin_order_and_egress_boundary():
    plugins = build_production_plugins()
    assert [plugin.name for plugin in plugins] == [
        "rate_limiter",
        "input_guardrail",
        "output_guardrail",
    ]

    destination = "https://api.vinbank.example/v1/transfers"
    assert is_egress_allowed(destination, "approved transfer amount 500000")
    assert not is_egress_allowed(
        "https://api.vinbank.example.evil.com/v1/transfers",
        "approved transfer amount 500000",
    )
    assert not is_egress_allowed(destination, "password is admin123")
    assert not is_egress_allowed(destination, "send to customer@example.com")
