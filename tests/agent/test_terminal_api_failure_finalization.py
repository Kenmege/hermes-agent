from types import SimpleNamespace

from agent.turn_finalizer import finalize_turn


class _Budget:
    remaining = 10
    max_total = 200
    used = 1


class _Agent:
    max_iterations = 200
    iteration_budget = _Budget()
    quiet_mode = True
    model = "gpt-5.5"
    provider = "openai-codex"
    base_url = "https://chatgpt.com/backend-api/codex/"
    session_id = "session-terminal-failure"
    session_input_tokens = 0
    session_output_tokens = 0
    session_cache_read_tokens = 0
    session_cache_write_tokens = 0
    session_reasoning_tokens = 0
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_total_tokens = 0
    session_estimated_cost_usd = 0.0
    session_cost_status = "unknown"
    session_cost_source = "unknown"
    context_compressor = SimpleNamespace(last_prompt_tokens=0)
    _tool_guardrail_halt_decision = None
    _response_was_previewed = False
    _interrupt_message = ""
    _skill_nudge_interval = 0
    _iters_since_skill = 0
    valid_tool_names = set()

    def __init__(self):
        self.persisted_messages = None
        self.cleanup_called = False

    def _save_trajectory(self, messages, user_message, completed):
        self.saved_completed = completed

    def _cleanup_task_resources(self, task_id):
        self.cleanup_called = True

    def _drop_trailing_empty_response_scaffolding(self, messages):
        return None

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return ""

    def clear_interrupt(self):
        self.interrupt_cleared = True

    def _sync_external_memory_for_turn(self, **kwargs):
        self.synced_external_memory = kwargs



def test_terminal_api_failure_result_keeps_error_metadata_and_finalizes():
    agent = _Agent()
    messages = [
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "API call failed after 3 retries: backend exploded"},
    ]

    result = finalize_turn(
        agent,
        final_response="API call failed after 3 retries: backend exploded",
        api_call_count=1,
        interrupted=False,
        failed=True,
        messages=messages,
        conversation_history=[],
        effective_task_id="default",
        turn_id="turn-1",
        user_message="continue",
        original_user_message="continue",
        _should_review_memory=False,
        _turn_exit_reason="api_failed_after_3_retries(transient)",
        error="backend exploded",
        failure_reason="transient",
    )

    assert result["completed"] is False
    assert result["failed"] is True
    assert result["error"] == "backend exploded"
    assert result["failure_reason"] == "transient"
    assert result["turn_exit_reason"] == "api_failed_after_3_retries(transient)"
    assert agent.cleanup_called is True
    assert agent.persisted_messages is not None
    assert agent.persisted_messages[-1]["role"] == "assistant"
    assert agent.persisted_messages[-1]["content"].startswith("API call failed after 3 retries")



def test_conversation_loop_routes_max_retry_failures_through_finalizer():
    # Regression guard for the Desktop/TUI wedge: max-retry API failures must
    # not early-return from the retry handler, because that skips finalize_turn
    # cleanup/diagnostics and leaves the stored session with an unresolved user
    # tail after compaction. Keep this as a source-contract test: exercising the
    # full live provider failure path would require real network credentials and
    # would be flaky.
    import inspect
    from agent import conversation_loop

    source = inspect.getsource(conversation_loop.run_conversation)
    start = source.index('reason="max_retries_exhausted"')
    end = source.index('# For rate limits', start)
    failure_block = source[start:end]

    assert "api_failed_after_{max_retries}_retries" in failure_block
    assert "messages.append({\"role\": \"assistant\", \"content\": final_response})" in failure_block
    assert "return {" not in failure_block
