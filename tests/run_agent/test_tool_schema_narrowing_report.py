from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "tool_schema_narrowing_report.py"
spec = importlib.util.spec_from_file_location("tool_schema_narrowing_report", MODULE_PATH)
assert spec and spec.loader
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def _row(**overrides):
    base = {
        "mode": "observe",
        "full_count": 100,
        "selected_count": 25,
        "tools_for_api_count": 100,
        "full_tokens": 40000,
        "selected_tokens": 10000,
        "saved_tokens": 30000,
        "cached_token_rate": 0.8,
        "cache_read_tokens": 80000,
        "cache_write_tokens": 0,
        "prompt_tokens": 100000,
        "input_tokens": 100000,
        "total_tokens": 100100,
        "latency_s": 10.0,
        "retry_count": 0,
        "rescue_count": 0,
    }
    base.update(overrides)
    return base


def test_load_jsonl_skips_bad_lines_and_summarizes_observe(tmp_path):
    log = tmp_path / "tool-schema.jsonl"
    rows = [_row(), _row(selected_count=30, saved_tokens=28000), {"not": "json serializable"}]
    log.write_text(
        json.dumps(rows[0])
        + "\nnot-json\n"
        + json.dumps(rows[1])
        + "\n"
        + json.dumps(rows[2])
        + "\n",
        encoding="utf-8",
    )

    loaded, invalid = report.load_jsonl_rows(log)
    summary = report.build_report(loaded, invalid_lines=invalid, min_enforce_rows=2)

    assert invalid == 1
    assert summary["totals"]["rows"] == 3
    observe = summary["modes"]["observe"]
    assert observe["rows"] == 2
    assert observe["avg_selected_count"] == 27.5
    assert observe["avg_tools_for_api_count"] == 100
    assert observe["total_saved_tokens_estimate"] == 58000
    assert summary["canary"]["status"] == "observe_only"


def test_enforce_canary_passes_when_cache_latency_and_tokens_improve():
    rows = [
        _row(mode="observe", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.8, latency_s=10.0),
        _row(mode="observe", input_tokens=98000, prompt_tokens=98000, cached_token_rate=0.82, latency_s=10.5),
        _row(mode="enforce", tools_for_api_count=25, input_tokens=60000, prompt_tokens=60000, cached_token_rate=0.83, latency_s=8.0),
        _row(mode="enforce", tools_for_api_count=25, input_tokens=59000, prompt_tokens=59000, cached_token_rate=0.82, latency_s=8.5),
    ]

    summary = report.build_report(rows, min_enforce_rows=2)

    assert summary["canary"]["status"] == "pass_candidate"
    assert summary["canary"]["input_token_delta_pct"] < 0
    assert summary["canary"]["total_input_token_delta_pct"] < 0
    assert summary["canary"]["total_processed_token_delta_pct"] < 0
    assert summary["canary"]["per_request_processed_token_delta_pct"] < 0
    assert summary["canary"]["latency_delta_pct"] < 0
    assert summary["canary"]["cache_rate_delta"] > -0.05


def test_enforce_canary_rolls_back_on_cache_degradation_even_with_schema_savings():
    rows = [
        _row(mode="observe", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.9, latency_s=10.0),
        _row(mode="observe", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.9, latency_s=10.0),
        _row(mode="enforce", tools_for_api_count=25, input_tokens=55000, prompt_tokens=55000, cached_token_rate=0.4, latency_s=9.0),
        _row(mode="enforce", tools_for_api_count=25, input_tokens=55000, prompt_tokens=55000, cached_token_rate=0.4, latency_s=9.0),
    ]

    summary = report.build_report(rows, min_enforce_rows=2)

    assert summary["canary"]["status"] == "rollback_to_observe"
    assert "cache_rate_drop" in summary["canary"]["reasons"]


def test_enforce_canary_rolls_back_on_latency_regression():
    rows = [
        _row(mode="observe", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.8, latency_s=10.0),
        _row(mode="observe", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.8, latency_s=10.0),
        _row(mode="enforce", input_tokens=50000, prompt_tokens=50000, cached_token_rate=0.8, latency_s=13.0),
        _row(mode="enforce", input_tokens=50000, prompt_tokens=50000, cached_token_rate=0.8, latency_s=13.0),
    ]

    summary = report.build_report(rows, min_enforce_rows=2, max_latency_increase_pct=0.15)

    assert summary["canary"]["status"] == "rollback_to_observe"
    assert "latency_regression" in summary["canary"]["reasons"]


def test_enforce_canary_rolls_back_on_retry_or_rescue_regression():
    rows = [
        _row(mode="observe", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.8),
        _row(mode="observe", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.8),
        _row(mode="enforce", input_tokens=50000, prompt_tokens=50000, cached_token_rate=0.8, rescue_count=1),
        _row(mode="enforce", input_tokens=50000, prompt_tokens=50000, cached_token_rate=0.8),
    ]

    summary = report.build_report(rows, min_enforce_rows=2, max_retry_rescue_rate=0.05)

    assert summary["canary"]["status"] == "rollback_to_observe"
    assert "retry_rescue_regression" in summary["canary"]["reasons"]


def test_enforce_canary_rolls_back_on_any_retry_or_rescue_regression_by_default():
    rows = [
        _row(mode="observe", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.8)
        for _ in range(20)
    ] + [
        _row(
            mode="enforce",
            input_tokens=50000,
            prompt_tokens=50000,
            cached_token_rate=0.8,
            rescue_count=1 if index == 0 else 0,
        )
        for index in range(20)
    ]

    summary = report.build_report(rows, min_enforce_rows=20)

    assert summary["canary"]["status"] == "rollback_to_observe"
    assert summary["canary"]["retry_rescue_rate_delta"] == 0.05
    assert "retry_rescue_regression" in summary["canary"]["reasons"]


def test_enforce_canary_rolls_back_without_processed_token_improvement():
    rows = [
        _row(mode="observe", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.8),
        _row(mode="observe", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.8),
        _row(mode="enforce", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.8),
        _row(mode="enforce", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.8),
    ]

    summary = report.build_report(rows, min_enforce_rows=2)

    assert summary["canary"]["status"] == "rollback_to_observe"
    assert "no_input_token_improvement" in summary["canary"]["reasons"]
    assert "no_processed_token_improvement" in summary["canary"]["reasons"]


def test_enforce_canary_rolls_back_when_totals_worsen_even_if_per_request_improves():
    rows = [
        _row(mode="observe", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.8),
        _row(mode="observe", input_tokens=100000, prompt_tokens=100000, cached_token_rate=0.8),
    ] + [
        _row(mode="enforce", input_tokens=50000, prompt_tokens=50000, cached_token_rate=0.8)
        for _ in range(5)
    ]

    summary = report.build_report(rows, min_enforce_rows=5)

    assert summary["canary"]["status"] == "rollback_to_observe"
    assert summary["canary"]["total_input_token_delta_pct"] > 0
    assert summary["canary"]["per_request_processed_token_delta_pct"] < 0
    assert "no_input_token_improvement" in summary["canary"]["reasons"]
    assert "no_processed_token_improvement" in summary["canary"]["reasons"]
