#!/usr/bin/env python3
"""Summarize Hermes dynamic tool-schema-narrowing JSONL telemetry.

The analyzer is intentionally read-only. It consumes compact audit rows emitted by
``agent.tool_schema_narrowing.log_tool_schema_narrowing_event`` and reports the
metrics needed before any enforce-mode canary: candidate schema savings,
provider cache behavior, latency, retry/rescue pressure, and processed tokens.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

DEFAULT_LOG_PATH = Path("/Users/kenmege/.hermes/logs/tool-schema-narrowing.jsonl")

NUMERIC_FIELDS = (
    "full_count",
    "selected_count",
    "tools_for_api_count",
    "full_tokens",
    "selected_tokens",
    "saved_tokens",
    "cached_token_rate",
    "cache_read_tokens",
    "cache_write_tokens",
    "prompt_tokens",
    "input_tokens",
    "total_tokens",
    "latency_s",
    "retry_count",
    "rescue_count",
)


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sum(rows: Iterable[dict[str, Any]], field: str) -> float:
    total = 0.0
    for row in rows:
        value = _as_float(row.get(field))
        if value is not None:
            total += value
    return total


def _avg(rows: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [_as_float(row.get(field)) for row in rows]
    clean = [value for value in values if value is not None]
    return round(mean(clean), 6) if clean else None


def _pct_delta(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return round((new - old) / old, 6)


def _percentile(rows: Iterable[dict[str, Any]], field: str, percentile: float) -> float | None:
    values = sorted(value for row in rows if (value := _as_float(row.get(field))) is not None)
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 6)
    rank = (len(values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return round(values[lower] * (1 - weight) + values[upper] * weight, 6)


def load_jsonl_rows(path: str | Path, *, limit: int | None = None) -> tuple[list[dict[str, Any]], int]:
    """Load JSONL telemetry rows, skipping malformed lines.

    Returns ``(rows, invalid_line_count)``. The function never raises for a
    missing file; callers get an empty dataset and can report that cleanly.
    """
    p = Path(path).expanduser()
    if not p.exists():
        return [], 0
    rows: list[dict[str, Any]] = []
    invalid = 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
        else:
            invalid += 1
    if limit is not None and limit > 0:
        rows = rows[-limit:]
    return rows, invalid


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one group of telemetry rows."""
    count = len(rows)
    selected = _avg(rows, "selected_count")
    full = _avg(rows, "full_count")
    selected_pct = None
    if selected is not None and full:
        selected_pct = round(selected / full, 6)

    prompt_tokens = _sum(rows, "prompt_tokens")
    input_tokens = _sum(rows, "input_tokens")
    cache_read = _sum(rows, "cache_read_tokens")
    cache_write = _sum(rows, "cache_write_tokens")
    processed_tokens = prompt_tokens or input_tokens
    uncached_input_tokens = max(0.0, processed_tokens - cache_read) if processed_tokens else None

    return {
        "rows": count,
        "avg_full_count": _avg(rows, "full_count"),
        "avg_selected_count": selected,
        "avg_tools_for_api_count": _avg(rows, "tools_for_api_count"),
        "avg_selected_to_full_ratio": selected_pct,
        "avg_full_tokens": _avg(rows, "full_tokens"),
        "avg_selected_tokens": _avg(rows, "selected_tokens"),
        "avg_saved_tokens": _avg(rows, "saved_tokens"),
        "p95_saved_tokens": _percentile(rows, "saved_tokens", 0.95),
        "total_saved_tokens_estimate": int(_sum(rows, "saved_tokens")),
        "avg_cached_token_rate": _avg(rows, "cached_token_rate"),
        "total_cache_read_tokens": int(cache_read),
        "total_cache_write_tokens": int(cache_write),
        "total_prompt_tokens": int(prompt_tokens),
        "total_input_tokens": int(input_tokens),
        "total_processed_tokens": int(processed_tokens),
        "total_uncached_input_tokens": int(uncached_input_tokens) if uncached_input_tokens is not None else None,
        "avg_prompt_tokens": _avg(rows, "prompt_tokens"),
        "avg_input_tokens": _avg(rows, "input_tokens"),
        "avg_total_tokens": _avg(rows, "total_tokens"),
        "avg_latency_s": _avg(rows, "latency_s"),
        "p95_latency_s": _percentile(rows, "latency_s", 0.95),
        "total_retry_count": int(_sum(rows, "retry_count")),
        "total_rescue_count": int(_sum(rows, "rescue_count")),
        "avg_retry_count": _avg(rows, "retry_count"),
        "avg_rescue_count": _avg(rows, "rescue_count"),
    }


def evaluate_canary(
    summaries: dict[str, dict[str, Any]],
    *,
    min_enforce_rows: int = 20,
    max_cache_rate_drop: float = 0.0,
    max_latency_increase_pct: float = 0.0,
    max_retry_rescue_rate: float = 0.0,
) -> dict[str, Any]:
    """Evaluate whether enforce canary evidence supports staying in enforce.

    The safe default is conservative: no enforce rows means observe-only, too
    few rows means insufficient sample, and any cache/latency/retry regression
    returns rollback_to_observe.
    """
    observe = summaries.get("observe")
    enforce = summaries.get("enforce")
    if not enforce or enforce.get("rows", 0) == 0:
        return {
            "status": "observe_only",
            "reasons": ["no_enforce_rows"],
            "message": "Only observe telemetry is present; do not enable enforce yet.",
        }
    if enforce.get("rows", 0) < min_enforce_rows:
        return {
            "status": "insufficient_enforce_sample",
            "reasons": ["enforce_rows_below_minimum"],
            "enforce_rows": enforce.get("rows", 0),
            "min_enforce_rows": min_enforce_rows,
            "message": "Enforce canary sample is too small for rollout decisions.",
        }
    if not observe or observe.get("rows", 0) == 0:
        return {
            "status": "missing_observe_baseline",
            "reasons": ["no_observe_rows"],
            "message": "Need observe baseline rows before judging enforce canary net impact.",
        }

    cache_delta = None
    if observe.get("avg_cached_token_rate") is not None and enforce.get("avg_cached_token_rate") is not None:
        cache_delta = round(enforce["avg_cached_token_rate"] - observe["avg_cached_token_rate"], 6)

    latency_delta_pct = _pct_delta(enforce.get("avg_latency_s"), observe.get("avg_latency_s"))
    input_delta_pct = _pct_delta(enforce.get("avg_input_tokens"), observe.get("avg_input_tokens"))
    prompt_delta_pct = _pct_delta(enforce.get("avg_prompt_tokens"), observe.get("avg_prompt_tokens"))
    total_input_delta_pct = _pct_delta(
        enforce.get("total_input_tokens"), observe.get("total_input_tokens")
    )
    total_processed_delta_pct = _pct_delta(
        enforce.get("total_processed_tokens"), observe.get("total_processed_tokens")
    )
    uncached_delta_pct = _pct_delta(
        enforce.get("total_uncached_input_tokens"), observe.get("total_uncached_input_tokens")
    )

    observe_rows = max(float(observe.get("rows", 0)), 1.0)
    enforce_rows = max(float(enforce.get("rows", 0)), 1.0)
    observe_processed_total = observe.get("total_processed_tokens")
    enforce_processed_total = enforce.get("total_processed_tokens")
    observe_processed_per_request = (
        observe_processed_total / observe_rows if observe_processed_total is not None else None
    )
    enforce_processed_per_request = (
        enforce_processed_total / enforce_rows if enforce_processed_total is not None else None
    )
    per_request_processed_delta_pct = _pct_delta(
        enforce_processed_per_request, observe_processed_per_request
    )
    observe_retry_rescue_rate = (
        (observe.get("total_retry_count", 0) + observe.get("total_rescue_count", 0))
        / observe_rows
    )
    retry_rescue_rate = (
        (enforce.get("total_retry_count", 0) + enforce.get("total_rescue_count", 0))
        / enforce_rows
    )
    retry_rescue_rate_delta = round(retry_rescue_rate - observe_retry_rescue_rate, 6)

    reasons: list[str] = []
    if cache_delta is None:
        reasons.append("missing_cache_rate")
    elif cache_delta < -max_cache_rate_drop:
        reasons.append("cache_rate_drop")

    if latency_delta_pct is None:
        reasons.append("missing_latency")
    elif latency_delta_pct > max_latency_increase_pct:
        reasons.append("latency_regression")

    if retry_rescue_rate_delta > max_retry_rescue_rate:
        reasons.append("retry_rescue_regression")

    # Provider token pressure must improve after cache impact. Kennedy's gate is
    # intentionally strict: total input tokens and total processed/prompt tokens
    # must be lower for the compared canary sample. If either total denominator
    # is missing or non-improving, enforcement rolls back to observe.
    if total_input_delta_pct is None:
        reasons.append("missing_total_input_tokens")
    elif total_input_delta_pct >= 0:
        reasons.append("no_input_token_improvement")

    if total_processed_delta_pct is None:
        reasons.append("missing_total_processed_tokens")
    elif total_processed_delta_pct >= 0:
        reasons.append("no_processed_token_improvement")

    status = "pass_candidate" if not reasons else "rollback_to_observe"
    return {
        "status": status,
        "reasons": reasons,
        "cache_rate_delta": cache_delta,
        "latency_delta_pct": latency_delta_pct,
        "input_token_delta_pct": input_delta_pct,
        "prompt_token_delta_pct": prompt_delta_pct,
        "total_input_token_delta_pct": total_input_delta_pct,
        "total_processed_token_delta_pct": total_processed_delta_pct,
        "per_request_processed_token_delta_pct": per_request_processed_delta_pct,
        "uncached_input_token_delta_pct": uncached_delta_pct,
        "retry_rescue_rate": round(retry_rescue_rate, 6),
        "observe_retry_rescue_rate": round(observe_retry_rescue_rate, 6),
        "retry_rescue_rate_delta": retry_rescue_rate_delta,
        "message": (
            "Enforce canary shows net improvement on token/cache/latency/retry gates."
            if status == "pass_candidate"
            else "Rollback to observe: enforce canary failed one or more cache-impact gates."
        ),
    }


def build_report(
    rows: list[dict[str, Any]],
    *,
    invalid_lines: int = 0,
    min_enforce_rows: int = 20,
    max_cache_rate_drop: float = 0.0,
    max_latency_increase_pct: float = 0.0,
    max_retry_rescue_rate: float = 0.0,
) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    route_counts: Counter[str] = Counter()
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        by_mode[mode].append(row)
        for route in row.get("matched_routes") or []:
            route_counts[str(route)] += 1

    mode_summaries = {mode: summarize_rows(group) for mode, group in sorted(by_mode.items())}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "rows": len(rows),
            "invalid_lines": invalid_lines,
            "modes": dict(Counter(str(row.get("mode") or "unknown") for row in rows)),
            "top_matched_routes": route_counts.most_common(12),
        },
        "modes": mode_summaries,
    }
    report["canary"] = evaluate_canary(
        mode_summaries,
        min_enforce_rows=min_enforce_rows,
        max_cache_rate_drop=max_cache_rate_drop,
        max_latency_increase_pct=max_latency_increase_pct,
        max_retry_rescue_rate=max_retry_rescue_rate,
    )
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Tool Schema Narrowing Telemetry Report",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Totals",
        "",
        f"- Rows: `{report['totals']['rows']}`",
        f"- Invalid JSONL lines skipped: `{report['totals']['invalid_lines']}`",
        f"- Modes: `{json.dumps(report['totals']['modes'], sort_keys=True)}`",
        f"- Top matched routes: `{json.dumps(report['totals']['top_matched_routes'])}`",
        "",
        "## Mode summaries",
        "",
    ]
    for mode, summary in report["modes"].items():
        lines.extend(
            [
                f"### {mode}",
                "",
                f"- Rows: `{summary['rows']}`",
                f"- Avg full tools / selected / sent: `{_fmt(summary['avg_full_count'])}` / `{_fmt(summary['avg_selected_count'])}` / `{_fmt(summary['avg_tools_for_api_count'])}`",
                f"- Avg selected/full ratio: `{_fmt(summary['avg_selected_to_full_ratio'])}`",
                f"- Avg saved schema tokens estimate: `{_fmt(summary['avg_saved_tokens'])}`",
                f"- Total saved schema tokens estimate: `{summary['total_saved_tokens_estimate']}`",
                f"- Avg cached-token rate: `{_fmt(summary['avg_cached_token_rate'])}`",
                f"- Total prompt/input/processed tokens: `{summary['total_prompt_tokens']}` / `{summary['total_input_tokens']}` / `{summary['total_processed_tokens']}`",
                f"- Total cache read/write tokens: `{summary['total_cache_read_tokens']}` / `{summary['total_cache_write_tokens']}`",
                f"- Total uncached input tokens: `{_fmt(summary['total_uncached_input_tokens'])}`",
                f"- Avg latency / p95 latency: `{_fmt(summary['avg_latency_s'])}` / `{_fmt(summary['p95_latency_s'])}` seconds",
                f"- Total retry/rescue count: `{summary['total_retry_count']}` / `{summary['total_rescue_count']}`",
                "",
            ]
        )
    canary = report["canary"]
    lines.extend(
        [
            "## Enforce canary gate",
            "",
            f"- Status: `{canary['status']}`",
            f"- Reasons: `{json.dumps(canary.get('reasons', []))}`",
            f"- Cache-rate delta: `{_fmt(canary.get('cache_rate_delta'))}`",
            f"- Latency delta pct: `{_fmt(canary.get('latency_delta_pct'))}`",
            f"- Input-token delta pct: `{_fmt(canary.get('input_token_delta_pct'))}`",
            f"- Prompt-token delta pct: `{_fmt(canary.get('prompt_token_delta_pct'))}`",
            f"- Total input-token delta pct: `{_fmt(canary.get('total_input_token_delta_pct'))}`",
            f"- Total processed-token delta pct: `{_fmt(canary.get('total_processed_token_delta_pct'))}`",
            f"- Per-request processed-token delta pct: `{_fmt(canary.get('per_request_processed_token_delta_pct'))}`",
            f"- Uncached-input delta pct: `{_fmt(canary.get('uncached_input_token_delta_pct'))}`",
            f"- Retry/rescue rate: `{_fmt(canary.get('retry_rescue_rate'))}`",
            f"- Observe retry/rescue rate: `{_fmt(canary.get('observe_retry_rescue_rate'))}`",
            f"- Retry/rescue rate delta: `{_fmt(canary.get('retry_rescue_rate_delta'))}`",
            f"- Message: {canary['message']}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", nargs="?", default=str(DEFAULT_LOG_PATH), help="JSONL audit log path")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    parser.add_argument("--limit", type=int, default=None, help="Only analyze the last N valid rows")
    parser.add_argument("--min-enforce-rows", type=int, default=20)
    parser.add_argument("--max-cache-rate-drop", type=float, default=0.0)
    parser.add_argument("--max-latency-increase-pct", type=float, default=0.0)
    parser.add_argument("--max-retry-rescue-rate", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, invalid = load_jsonl_rows(args.log_path, limit=args.limit)
    report = build_report(
        rows,
        invalid_lines=invalid,
        min_enforce_rows=args.min_enforce_rows,
        max_cache_rate_drop=args.max_cache_rate_drop,
        max_latency_increase_pct=args.max_latency_increase_pct,
        max_retry_rescue_rate=args.max_retry_rescue_rate,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
