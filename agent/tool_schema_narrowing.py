"""Request-time tool schema narrowing helpers.

This module intentionally has no side effects on the canonical ``agent.tools``
registry.  It computes an effective per-request tools array and telemetry so
observe mode can measure candidate savings before any enforce rollout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

VALID_MODES = {"off", "observe", "enforce"}

DEFAULT_ALWAYS_INCLUDE: tuple[str, ...] = (
    "clarify",
    "todo",
    "skill_view",
    "skills_list",
    "session_search",
    "delegate_task",
    "terminal",
    "process",
    "read_file",
    "search_files",
    "write_file",
    "patch",
    "execute_code",
)

SAFETY_FLOOR_TOOLS: tuple[str, ...] = (
    "todo",
    "clarify",
    "skill_view",
    "skills_list",
    "session_search",
    "terminal",
    "process",
    "read_file",
    "search_files",
    "write_file",
    "patch",
    "execute_code",
    "delegate_task",
)

TOOLSET_TO_TOOLS: dict[str, tuple[str, ...]] = {
    "file": ("read_file", "search_files", "write_file", "patch"),
    "terminal": ("terminal", "process"),
    "code_execution": ("execute_code",),
    "web": ("web_search", "web_extract", "x_search"),
    "browser": (
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_snapshot",
        "browser_scroll",
        "browser_press",
        "browser_back",
        "browser_console",
        "browser_vision",
        "browser_cdp",
        "browser_dialog",
        "browser_get_images",
    ),
    "computer_use": ("computer_use",),
    "messaging": ("send_message",),
    "vision": ("vision_analyze", "browser_vision"),
    "image_gen": ("image_generate",),
    "video": ("video_analyze",),
    "video_gen": ("video_generate",),
    "tts": ("text_to_speech",),
    "memory": ("memory", "brv_query", "brv_curate", "brv_status"),
}

MCP_SERVER_PREFIXES: dict[str, tuple[str, ...]] = {
    "linear": ("mcp_linear_",),
    "browser-engine": ("mcp_browser_engine_",),
    "browser_engine": ("mcp_browser_engine_",),
    "tinyfish": ("mcp_tinyfish_",),
    "context-mode": ("mcp_context_mode_",),
    "context_mode": ("mcp_context_mode_",),
    "claude-cli": ("mcp_claude_cli_",),
    "claude_cli": ("mcp_claude_cli_",),
}

PREFIX_TO_ROUTE: dict[str, str] = {
    "mcp_linear_": "linear",
    "mcp_tinyfish_": "web",
    "mcp_browser_engine_": "browser",
    "mcp_context_mode_": "context_mode",
    "browser_": "browser",
}

DEFAULT_ROUTE_TRIGGERS: dict[str, dict[str, Any]] = {
    "file": {
        "patterns": ["file", "read", "write", "patch", "edit", "diff", "repo", "test", "pytest", "git"],
        "include_toolsets": ["file", "terminal", "code_execution"],
    },
    "web": {
        "patterns": ["http://", "https://", "search", "latest", "current", "docs", "pricing", "web"],
        "include_toolsets": ["web"],
        "include_mcp_servers": ["tinyfish"],
    },
    "browser": {
        "patterns": ["browser", "click", "login", "chrome", "screenshot", "ui", "web page"],
        "include_toolsets": ["browser", "computer_use"],
        "include_mcp_servers": ["browser-engine"],
    },
    "linear": {
        "patterns": ["linear", "issue", "project", "milestone", "cycle"],
        "include_mcp_servers": ["linear"],
    },
    "context_mode": {
        "patterns": ["context-mode", "ctx_", "index", "bm25", "knowledge base"],
        "include_mcp_servers": ["context-mode"],
    },
    "messaging": {
        "patterns": ["send", "telegram", "discord", "message", "notify"],
        "include_toolsets": ["messaging"],
    },
    "media": {
        "patterns": ["image", "video", "audio", "tts", "vision", "screenshot"],
        "include_toolsets": ["vision", "image_gen", "video", "video_gen", "tts"],
    },
}

UNAVAILABLE_TOOL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bI (do not|don't) have (access to|the ability to use)\b",
        r"\b(no|not) available tool\b",
        r"\bI (can't|cannot) (browse|open|read files|run commands|access)\b",
        r"\bwithout (access to|the ability to use) (tools?|browsing|files?|commands?)\b",
    )
)


@dataclass(frozen=True)
class ToolSchemaNarrowingConfig:
    enabled: bool = False
    mode: str = "observe"  # off | observe | enforce
    max_visible_tools: int = 45
    min_visible_tools: int = 18
    max_schema_tokens: int = 18_000
    always_include: tuple[str, ...] = DEFAULT_ALWAYS_INCLUDE
    fallback_full_tools_on_error: bool = True
    retry_full_tools_on_unavailable_tool_response: bool = True
    include_recent_tool_families: bool = True
    include_loaded_skill_families: bool = True
    include_mcp_servers_on_trigger: bool = True
    route_triggers: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_ROUTE_TRIGGERS))
    log_path: str = ""


@dataclass
class ToolSchemaNarrowingDecision:
    mode: str
    active: bool
    tools_for_api: list[dict[str, Any]]
    full_count: int
    selected_count: int
    full_tokens: int
    selected_tokens: int
    saved_tokens: int
    selected_names: list[str]
    matched_routes: list[str]
    reason: str
    error: str | None = None

    def to_log_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Tool schemas themselves can be large. Log names/counts only.
        data.pop("tools_for_api", None)
        return data


def tool_name(tool: Mapping[str, Any]) -> str:
    return str((tool.get("function") or {}).get("name") or "")


def estimate_tool_tokens(tools: Iterable[Mapping[str, Any]]) -> int:
    """Rough relative schema-token estimate, not a billing-token meter.

    Enforcement canary analysis must prefer provider usage fields from JSONL
    (input/prompt/cache tokens) when deciding whether prompt-cache degradation
    outweighs schema savings.
    """
    chars = 0
    for tool in tools:
        try:
            chars += len(json.dumps(tool, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
        except Exception:
            chars += len(str(tool))
    return chars // 4


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _as_int(value: Any, default: int, minimum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    return result


def _as_tuple(value: Any, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if str(item))
    return default


def _merge_route_triggers(raw: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {key: dict(value) for key, value in DEFAULT_ROUTE_TRIGGERS.items()}
    if not isinstance(raw, Mapping):
        return merged
    for route, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        current = dict(merged.get(str(route), {}))
        current.update(dict(value))
        merged[str(route)] = current
    return merged


def load_tool_schema_narrowing_config(raw_config: Mapping[str, Any] | None) -> ToolSchemaNarrowingConfig:
    """Load narrowing config from the root Hermes config or the subsection.

    The default is disabled/off unless ``tools.dynamic_schema_narrowing`` (or a
    direct subsection) sets ``enabled: true``. This keeps a code deploy inert
    until observe-mode rollout explicitly enables logging.
    """
    raw_config = raw_config or {}
    if not isinstance(raw_config, Mapping):
        raw_config = {}

    if "dynamic_schema_narrowing" in raw_config:
        block = raw_config.get("dynamic_schema_narrowing") or {}
    elif "tools" in raw_config:
        tools_cfg = raw_config.get("tools") or {}
        block = tools_cfg.get("dynamic_schema_narrowing") if isinstance(tools_cfg, Mapping) else {}
    else:
        block = raw_config

    if not isinstance(block, Mapping):
        block = {}

    mode = str(block.get("mode", "observe") or "observe").strip().lower()
    if mode not in VALID_MODES:
        logger.warning("Invalid dynamic_schema_narrowing.mode=%r; falling back to observe", mode)
        mode = "observe"

    enabled = _as_bool(block.get("enabled"), False)
    if mode == "off":
        enabled = False

    always_include = _as_tuple(block.get("always_include"), DEFAULT_ALWAYS_INCLUDE)
    if not always_include:
        always_include = DEFAULT_ALWAYS_INCLUDE

    return ToolSchemaNarrowingConfig(
        enabled=enabled,
        mode=mode,
        max_visible_tools=_as_int(block.get("max_visible_tools"), 45, minimum=1),
        min_visible_tools=_as_int(block.get("min_visible_tools"), 18, minimum=0),
        max_schema_tokens=_as_int(block.get("max_schema_tokens"), 18_000, minimum=0),
        always_include=always_include,
        fallback_full_tools_on_error=_as_bool(block.get("fallback_full_tools_on_error"), True),
        retry_full_tools_on_unavailable_tool_response=_as_bool(
            block.get("retry_full_tools_on_unavailable_tool_response"), True
        ),
        include_recent_tool_families=_as_bool(block.get("include_recent_tool_families"), True),
        include_loaded_skill_families=_as_bool(block.get("include_loaded_skill_families"), True),
        include_mcp_servers_on_trigger=_as_bool(block.get("include_mcp_servers_on_trigger"), True),
        route_triggers=_merge_route_triggers(block.get("route_triggers")),
        log_path=str(block.get("log_path") or ""),
    )


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, Mapping):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _collect_routing_text(api_messages: list[dict[str, Any]], original_user_message: str, recent_tool_names: list[str] | None) -> str:
    parts = [original_user_message or ""]
    for msg in api_messages[-8:]:
        if msg.get("role") in {"user", "assistant"}:
            parts.append(_text_from_content(msg.get("content")))
        for call in msg.get("tool_calls") or []:
            if isinstance(call, Mapping):
                name = str((call.get("function") or {}).get("name") or "")
                if name:
                    parts.append(name)
    if recent_tool_names:
        parts.extend(str(name) for name in recent_tool_names)
    return "\n".join(parts).lower()


def _tool_call_names(api_messages: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for msg in api_messages:
        for call in msg.get("tool_calls") or []:
            if isinstance(call, Mapping):
                name = str((call.get("function") or {}).get("name") or "")
                if name:
                    names.add(name)
    return names


def _route_for_tool_name(name: str) -> str | None:
    for prefix, route in PREFIX_TO_ROUTE.items():
        if name.startswith(prefix):
            return route
    if name in TOOLSET_TO_TOOLS.get("web", ()):
        return "web"
    if name in TOOLSET_TO_TOOLS.get("browser", ()) or name in TOOLSET_TO_TOOLS.get("computer_use", ()):
        return "browser"
    if name in TOOLSET_TO_TOOLS.get("file", ()) or name in TOOLSET_TO_TOOLS.get("terminal", ()) or name in TOOLSET_TO_TOOLS.get("code_execution", ()):
        return "file"
    if name in TOOLSET_TO_TOOLS.get("messaging", ()):
        return "messaging"
    if name in TOOLSET_TO_TOOLS.get("vision", ()) or name in TOOLSET_TO_TOOLS.get("image_gen", ()) or name in TOOLSET_TO_TOOLS.get("video", ()) or name in TOOLSET_TO_TOOLS.get("video_gen", ()) or name in TOOLSET_TO_TOOLS.get("tts", ()):
        return "media"
    return None


def _add_name(selected: set[str], priorities: dict[str, int], available: set[str], name: str, priority: int) -> None:
    if name in available:
        selected.add(name)
        priorities[name] = max(priorities.get(name, 0), priority)


def _add_prefix(selected: set[str], priorities: dict[str, int], names: Iterable[str], prefix: str, priority: int) -> None:
    for name in names:
        if name.startswith(prefix):
            selected.add(name)
            priorities[name] = max(priorities.get(name, 0), priority)


def _add_toolset(selected: set[str], priorities: dict[str, int], available: set[str], toolset: str, priority: int) -> None:
    for name in TOOLSET_TO_TOOLS.get(toolset, ()):
        _add_name(selected, priorities, available, name, priority)


def _add_mcp_server(selected: set[str], priorities: dict[str, int], all_names: Iterable[str], server: str, priority: int) -> None:
    for prefix in MCP_SERVER_PREFIXES.get(server, (f"mcp_{server.replace('-', '_')}_",)):
        _add_prefix(selected, priorities, all_names, prefix, priority)


def _matches_any_pattern(text: str, patterns: Iterable[Any]) -> bool:
    for pattern in patterns or []:
        pat = str(pattern or "").strip().lower()
        if not pat:
            continue
        if pat.startswith("re:"):
            try:
                if re.search(pat[3:], text, flags=re.IGNORECASE):
                    return True
            except re.error:
                continue
        elif pat in text:
            return True
    return False


def _ordered_tools(all_tools: list[dict[str, Any]], names: set[str]) -> list[dict[str, Any]]:
    return [tool for tool in all_tools if tool_name(tool) in names]


def _trim_selection(
    *,
    all_tools: list[dict[str, Any]],
    selected: set[str],
    priorities: dict[str, int],
    always: set[str],
    max_visible_tools: int,
    max_schema_tokens: int,
) -> set[str]:
    if not selected:
        return selected

    def selected_tools() -> list[dict[str, Any]]:
        return _ordered_tools(all_tools, selected)

    # Drop low-priority optional tools first, preserving original tool order for
    # the final payload.  Always-include and unresolved/recent exact tools are
    # safety-critical; prefer exceeding caps to silently removing them.
    while max_visible_tools and len(selected) > max_visible_tools:
        removable = [name for name in selected if name not in always and priorities.get(name, 0) < 90]
        if not removable:
            break
        victim = min(removable, key=lambda n: (priorities.get(n, 0), n))
        selected.remove(victim)

    while max_schema_tokens and estimate_tool_tokens(selected_tools()) > max_schema_tokens:
        removable = [name for name in selected if name not in always and priorities.get(name, 0) < 90]
        if not removable:
            break
        victim = min(removable, key=lambda n: (priorities.get(n, 0), n))
        selected.remove(victim)

    return selected


def _full_decision(
    *,
    mode: str,
    active: bool,
    all_tools: list[dict[str, Any]],
    full_tokens: int,
    reason: str,
    error: str | None = None,
) -> ToolSchemaNarrowingDecision:
    names = [tool_name(tool) for tool in all_tools]
    return ToolSchemaNarrowingDecision(
        mode=mode,
        active=active,
        tools_for_api=all_tools,
        full_count=len(all_tools),
        selected_count=len(all_tools),
        full_tokens=full_tokens,
        selected_tokens=full_tokens,
        saved_tokens=0,
        selected_names=names,
        matched_routes=[],
        reason=reason,
        error=error,
    )


def choose_tools_for_request(
    *,
    config: ToolSchemaNarrowingConfig,
    all_tools: list[dict[str, Any]],
    api_messages: list[dict[str, Any]],
    original_user_message: str,
    recent_tool_names: list[str] | None = None,
    force_full_tools: bool = False,
) -> ToolSchemaNarrowingDecision:
    """Choose the provider-visible tool schemas for one request.

    ``observe`` mode returns the full tool list to the API but reports the
    narrowed candidate set for measurement. ``enforce`` mode returns the
    candidate set. ``off``/disabled and all errors fail open to full tools.
    """
    full_tokens = estimate_tool_tokens(all_tools)
    mode = (config.mode or "observe").lower()

    if force_full_tools:
        return _full_decision(
            mode=mode,
            active=bool(config.enabled and mode in {"observe", "enforce"}),
            all_tools=all_tools,
            full_tokens=full_tokens,
            reason="forced_full_tools",
        )

    if not config.enabled or mode == "off":
        return _full_decision(
            mode="off",
            active=False,
            all_tools=all_tools,
            full_tokens=full_tokens,
            reason="disabled",
        )

    if mode not in {"observe", "enforce"}:
        return _full_decision(
            mode="off",
            active=False,
            all_tools=all_tools,
            full_tokens=full_tokens,
            reason=f"invalid_mode:{mode}",
        )

    try:
        all_names = [tool_name(tool) for tool in all_tools]
        available = set(all_names)
        selected: set[str] = set()
        priorities: dict[str, int] = {}
        matched_routes: list[str] = []
        always = {name for name in config.always_include if name in available}

        for name in config.always_include:
            _add_name(selected, priorities, available, name, 100)

        routing_text = _collect_routing_text(api_messages, original_user_message, recent_tool_names)

        for route, route_cfg in (config.route_triggers or {}).items():
            if not isinstance(route_cfg, Mapping):
                continue
            if not _matches_any_pattern(routing_text, route_cfg.get("patterns") or ()):
                continue
            route = str(route)
            matched_routes.append(route)
            for toolset in route_cfg.get("include_toolsets") or ():
                _add_toolset(selected, priorities, available, str(toolset), 80)
            if config.include_mcp_servers_on_trigger:
                for server in route_cfg.get("include_mcp_servers") or ():
                    _add_mcp_server(selected, priorities, all_names, str(server), 80)

            # Heuristic route families catch MCPs and direct tool prefixes even
            # when the config names only the logical route.
            for name in all_names:
                if _route_for_tool_name(name) == route:
                    _add_name(selected, priorities, available, name, 75)

        # Exact tool-name mentions in the current turn are high-signal.
        for name in all_names:
            if name and name.lower() in routing_text:
                _add_name(selected, priorities, available, name, 92)

        # Preserve tool calls already present in the API transcript.  Some
        # providers validate replayed assistant tool_call names against the
        # current tools array.
        for name in _tool_call_names(api_messages):
            _add_name(selected, priorities, available, name, 95)
            route = _route_for_tool_name(name)
            if route and config.include_recent_tool_families:
                matched_routes.append(route)
                for other in all_names:
                    if _route_for_tool_name(other) == route:
                        _add_name(selected, priorities, available, other, 76)

        if config.include_recent_tool_families and recent_tool_names:
            for recent in recent_tool_names:
                recent = str(recent)
                _add_name(selected, priorities, available, recent, 95)
                route = _route_for_tool_name(recent)
                if route:
                    matched_routes.append(route)
                    for other in all_names:
                        if _route_for_tool_name(other) == route:
                            _add_name(selected, priorities, available, other, 76)

        # Safety floor: keep enough operator/supervisor tools visible for
        # common local work even when routing text is too sparse.
        for name in SAFETY_FLOOR_TOOLS:
            if len(selected) >= config.min_visible_tools:
                break
            _add_name(selected, priorities, available, name, 50)

        if not selected:
            return _full_decision(
                mode=mode,
                active=True,
                all_tools=all_tools,
                full_tokens=full_tokens,
                reason="empty_candidate_fail_open",
            )

        selected = _trim_selection(
            all_tools=all_tools,
            selected=selected,
            priorities=priorities,
            always=always,
            max_visible_tools=config.max_visible_tools,
            max_schema_tokens=config.max_schema_tokens,
        )
        selected_tools = _ordered_tools(all_tools, selected)
        selected_names = [tool_name(tool) for tool in selected_tools]
        selected_tokens = estimate_tool_tokens(selected_tools)
        selected_count = len(selected_tools)

        if selected_count == 0:
            return _full_decision(
                mode=mode,
                active=True,
                all_tools=all_tools,
                full_tokens=full_tokens,
                reason="trimmed_empty_fail_open",
            )

        tools_for_api = all_tools if mode == "observe" else selected_tools
        matched_routes = sorted(set(matched_routes))
        reason = "matched:" + ",".join(matched_routes) if matched_routes else "safety_floor"
        return ToolSchemaNarrowingDecision(
            mode=mode,
            active=True,
            tools_for_api=tools_for_api,
            full_count=len(all_tools),
            selected_count=selected_count,
            full_tokens=full_tokens,
            selected_tokens=selected_tokens,
            saved_tokens=max(0, full_tokens - selected_tokens),
            selected_names=selected_names,
            matched_routes=matched_routes,
            reason=reason,
        )
    except Exception as exc:
        if not config.fallback_full_tools_on_error:
            raise
        logger.warning("Tool schema narrowing failed open: %s", exc)
        return _full_decision(
            mode=mode,
            active=True,
            all_tools=all_tools,
            full_tokens=full_tokens,
            reason="exception_fail_open",
            error=str(exc),
        )


def should_retry_with_full_tools(
    decision: ToolSchemaNarrowingDecision | None,
    assistant_text: str,
    *,
    tool_calls_made: bool = False,
    already_rescued: bool = False,
) -> bool:
    if decision is None:
        return False
    if already_rescued or tool_calls_made:
        return False
    if decision.mode != "enforce" or not decision.active:
        return False
    if decision.selected_count >= decision.full_count:
        return False
    text = assistant_text or ""
    return any(pattern.search(text) for pattern in UNAVAILABLE_TOOL_PATTERNS)


def _coerce_usage_dict(provider_usage: Any) -> dict[str, Any]:
    if provider_usage is None:
        return {}
    if isinstance(provider_usage, Mapping):
        return dict(provider_usage)
    keys = (
        "prompt_tokens",
        "input_tokens",
        "output_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    )
    return {key: getattr(provider_usage, key) for key in keys if hasattr(provider_usage, key)}


def log_tool_schema_narrowing_event(
    decision: ToolSchemaNarrowingDecision | None,
    *,
    log_path: str,
    session_id: str | None = None,
    api_call_count: int | None = None,
    latency_s: float | None = None,
    retry_count: int | None = None,
    rescue_count: int | None = None,
    provider_usage: Any = None,
    provider: str | None = None,
    model: str | None = None,
    marker: str | None = None,
) -> None:
    """Append a compact JSONL audit row.

    The row intentionally excludes prompts, message bodies, tool arguments, and
    full schemas. It is safe to leave on in observe mode.
    """
    if decision is None or not log_path:
        return
    try:
        usage = _coerce_usage_dict(provider_usage)
        row: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id or "",
            "api_call_count": api_call_count,
            "mode": decision.mode,
            "active": decision.active,
            "full_count": decision.full_count,
            "selected_count": decision.selected_count,
            "tools_for_api_count": len(decision.tools_for_api or []),
            "full_tokens": decision.full_tokens,
            "selected_tokens": decision.selected_tokens,
            "saved_tokens": decision.saved_tokens,
            "matched_routes": list(decision.matched_routes),
            "selected_names": list(decision.selected_names),
            "reason": decision.reason,
            "error": decision.error,
            "latency_s": round(float(latency_s), 3) if latency_s is not None else None,
            "retry_count": retry_count,
            "rescue_count": rescue_count,
            "provider": provider or "",
            "model": model or "",
        }
        if marker:
            row["marker"] = marker
        for key in (
            "prompt_tokens",
            "input_tokens",
            "output_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
        ):
            if key in usage and usage[key] is not None:
                row[key] = usage[key]
        prompt_tokens = row.get("prompt_tokens") or row.get("input_tokens")
        cache_read = row.get("cache_read_tokens")
        # Cache-hit rate is emitted only when the provider reports a prompt or
        # input denominator. Some providers expose cache-read buckets without a
        # usable denominator; leave the ratio absent instead of inventing one.
        if prompt_tokens:
            row["cached_token_rate"] = round(float(cache_read or 0) / float(prompt_tokens), 6)
        path = Path(log_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as exc:
        logger.debug("tool schema narrowing audit log failed: %s", exc)
