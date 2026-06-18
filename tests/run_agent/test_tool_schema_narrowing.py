import json
from pathlib import Path



def _tool(name: str, description: str | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }



def _tools(*names: str) -> list[dict]:
    return [_tool(name) for name in names]



def _config(mode: str = "observe", **overrides):
    from agent.tool_schema_narrowing import ToolSchemaNarrowingConfig

    values = {
        "enabled": True,
        "mode": mode,
        "always_include": ("todo", "terminal"),
        "min_visible_tools": 2,
        "max_visible_tools": 8,
        "max_schema_tokens": 100_000,
        "route_triggers": {
            "linear": {
                "patterns": ["linear", "issue"],
                "include_mcp_servers": ["linear"],
            },
            "browser": {
                "patterns": ["browser", "chrome", "click"],
                "include_mcp_servers": ["browser-engine"],
            },
            "file": {
                "patterns": ["file", "read", "repo", "pytest"],
                "include_toolsets": ["file", "terminal", "code_execution"],
            },
        },
    }
    values.update(overrides)
    return ToolSchemaNarrowingConfig(**values)



def test_mode_off_returns_full_tools():
    from agent.tool_schema_narrowing import choose_tools_for_request

    all_tools = _tools("todo", "terminal", "mcp_linear_get_issue")
    decision = choose_tools_for_request(
        config=_config("off"),
        all_tools=all_tools,
        api_messages=[{"role": "user", "content": "open a Linear issue"}],
        original_user_message="open a Linear issue",
    )

    assert decision.tools_for_api == all_tools
    assert decision.selected_names == ["todo", "terminal", "mcp_linear_get_issue"]
    assert decision.saved_tokens == 0
    assert decision.active is False



def test_observe_returns_full_tools_but_reports_candidate_savings():
    from agent.tool_schema_narrowing import choose_tools_for_request

    all_tools = _tools(
        "todo",
        "terminal",
        "read_file",
        "search_files",
        "mcp_linear_get_issue",
        "mcp_browser_engine_browser_engine_status",
        "mcp_tinyfish_search",
    )
    decision = choose_tools_for_request(
        config=_config("observe"),
        all_tools=all_tools,
        api_messages=[{"role": "user", "content": "read this repo file"}],
        original_user_message="read this repo file",
    )

    assert decision.tools_for_api == all_tools
    assert set(decision.selected_names) >= {"todo", "terminal", "read_file", "search_files"}
    assert "mcp_tinyfish_search" not in decision.selected_names
    assert decision.selected_count < decision.full_count
    assert decision.saved_tokens > 0
    assert decision.active is True



def test_enforce_returns_narrowed_tools_without_mutating_full_list():
    from agent.tool_schema_narrowing import choose_tools_for_request, tool_name

    all_tools = _tools(
        "todo",
        "terminal",
        "read_file",
        "search_files",
        "mcp_linear_get_issue",
        "mcp_tinyfish_search",
    )
    before_names = [tool_name(t) for t in all_tools]
    decision = choose_tools_for_request(
        config=_config("enforce"),
        all_tools=all_tools,
        api_messages=[{"role": "user", "content": "read a file and run pytest"}],
        original_user_message="read a file and run pytest",
    )

    assert [tool_name(t) for t in all_tools] == before_names
    assert decision.tools_for_api != all_tools
    assert [tool_name(t) for t in decision.tools_for_api] == decision.selected_names
    assert "mcp_tinyfish_search" not in decision.selected_names



def test_linear_trigger_includes_linear_mcp_tools():
    from agent.tool_schema_narrowing import choose_tools_for_request

    all_tools = _tools("todo", "terminal", "mcp_linear_get_issue", "mcp_linear_list_issues", "browser_navigate")
    decision = choose_tools_for_request(
        config=_config("enforce"),
        all_tools=all_tools,
        api_messages=[{"role": "user", "content": "check Linear issue KEN-1"}],
        original_user_message="check Linear issue KEN-1",
    )

    assert "linear" in decision.matched_routes
    assert "mcp_linear_get_issue" in decision.selected_names
    assert "mcp_linear_list_issues" in decision.selected_names



def test_browser_trigger_prefers_browser_engine_family():
    from agent.tool_schema_narrowing import choose_tools_for_request

    all_tools = _tools(
        "todo",
        "terminal",
        "browser_navigate",
        "mcp_browser_engine_browser_engine_status",
        "mcp_tinyfish_run_web_automation",
    )
    decision = choose_tools_for_request(
        config=_config("enforce"),
        all_tools=all_tools,
        api_messages=[{"role": "user", "content": "check Browser Engine Chrome status"}],
        original_user_message="check Browser Engine Chrome status",
    )

    assert "browser" in decision.matched_routes
    assert "mcp_browser_engine_browser_engine_status" in decision.selected_names
    assert "mcp_tinyfish_run_web_automation" not in decision.selected_names



def test_unresolved_tool_call_name_is_preserved():
    from agent.tool_schema_narrowing import choose_tools_for_request

    all_tools = _tools("todo", "terminal", "mcp_context_mode_ctx_search", "mcp_tinyfish_search")
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "mcp_context_mode_ctx_search", "arguments": "{}"}}]},
    ]
    decision = choose_tools_for_request(
        config=_config("enforce"),
        all_tools=all_tools,
        api_messages=messages,
        original_user_message="continue",
    )

    assert "mcp_context_mode_ctx_search" in decision.selected_names



def test_force_full_tools_keeps_full_registry_for_rescue_retry():
    from agent.tool_schema_narrowing import choose_tools_for_request

    all_tools = _tools("todo", "terminal", "read_file", "mcp_tinyfish_search")
    decision = choose_tools_for_request(
        config=_config("enforce"),
        all_tools=all_tools,
        api_messages=[{"role": "user", "content": "read a file"}],
        original_user_message="read a file",
        force_full_tools=True,
    )

    assert decision.tools_for_api == all_tools
    assert decision.selected_count == decision.full_count
    assert decision.saved_tokens == 0
    assert decision.reason == "forced_full_tools"



def test_unavailable_tool_response_detection_is_conservative():
    from agent.tool_schema_narrowing import ToolSchemaNarrowingDecision, should_retry_with_full_tools

    decision = ToolSchemaNarrowingDecision(
        mode="enforce",
        active=True,
        tools_for_api=[],
        full_count=10,
        selected_count=4,
        full_tokens=1000,
        selected_tokens=400,
        saved_tokens=600,
        selected_names=[],
        matched_routes=[],
        reason="matched",
    )

    assert should_retry_with_full_tools(decision, "I don't have access to browse the web.") is True
    assert should_retry_with_full_tools(decision, "Task complete.") is False
    assert should_retry_with_full_tools(decision, "I don't have access to browse.", tool_calls_made=True) is False
    assert should_retry_with_full_tools(decision, "I don't have access to browse.", already_rescued=True) is False



def test_jsonl_logging_omits_prompt_content_and_records_cache_fields(tmp_path: Path):
    from agent.tool_schema_narrowing import ToolSchemaNarrowingDecision, log_tool_schema_narrowing_event

    log_path = tmp_path / "tool-schema-narrowing.jsonl"
    decision = ToolSchemaNarrowingDecision(
        mode="observe",
        active=True,
        tools_for_api=[],
        full_count=20,
        selected_count=5,
        full_tokens=10_000,
        selected_tokens=2_000,
        saved_tokens=8_000,
        selected_names=["terminal", "read_file"],
        matched_routes=["file"],
        reason="matched:file",
    )

    log_tool_schema_narrowing_event(
        decision,
        log_path=str(log_path),
        session_id="sess-1",
        api_call_count=3,
        latency_s=1.25,
        retry_count=1,
        rescue_count=0,
        provider_usage={
            "prompt_tokens": 100,
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "cache_read_tokens": 60,
            "cache_write_tokens": 20,
        },
    )

    row = json.loads(log_path.read_text().strip())
    assert row["session_id"] == "sess-1"
    assert row["api_call_count"] == 3
    assert row["saved_tokens"] == 8_000
    assert row["cache_read_tokens"] == 60
    assert row["cached_token_rate"] == 0.6
    assert "content" not in row
    assert "messages" not in row
