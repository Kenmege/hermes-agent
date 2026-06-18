from types import SimpleNamespace

from agent.conversation_loop import _maybe_compress_before_api_call


class FakeCompressor:
    def __init__(self, *, defer=False, threshold=100):
        self.protect_first_n = 1
        self.protect_last_n = 1
        self.threshold_tokens = threshold
        self.context_length = 200
        self.last_real_prompt_tokens = 0
        self.defer = defer

    def should_defer_preflight_to_real_usage(self, tokens):
        return self.defer

    def should_compress(self, tokens):
        return tokens >= self.threshold_tokens


def _agent(*, defer=False, compression_enabled=True):
    statuses = []

    def compress(messages, system_message, approx_tokens, task_id):
        # Shrink enough to prove the guard restarts before a provider call.
        return [messages[0], {"role": "assistant", "content": "[summary]"}, messages[-1]], "compressed system"

    agent = SimpleNamespace(
        compression_enabled=compression_enabled,
        context_compressor=FakeCompressor(defer=defer),
        model="test-model",
        _empty_content_retries=2,
        _thinking_prefill_retries=2,
        _last_content_with_tools="stale",
        _last_content_tools_all_housekeeping=True,
        _mute_post_response=True,
        _buffer_status=statuses.append,
        _emit_status=statuses.append,
        _compress_context=compress,
    )
    agent.statuses = statuses
    return agent


def _messages():
    return [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "call tool"},
        {"role": "tool", "tool_call_id": "tc1", "content": "large result"},
        {"role": "assistant", "content": "next"},
    ]


def test_in_turn_guard_compresses_before_next_api_request():
    agent = _agent()
    history = list(_messages())

    result = _maybe_compress_before_api_call(
        agent,
        history,
        system_message="system",
        active_system_prompt="active system",
        conversation_history=history,
        approx_request_tokens=150,
        effective_task_id="task-1",
        compression_attempts=0,
        max_compression_attempts=3,
        api_call_count=7,
    )

    assert result["compressed"] is True
    assert result["exhausted"] is False
    assert result["compression_attempts"] == 1
    assert result["conversation_history"] is None
    assert result["active_system_prompt"] == "compressed system"
    assert len(result["messages"]) == 3
    assert any("In-turn compression before API call #7" in s for s in agent.statuses)
    assert agent._empty_content_retries == 0
    assert agent._thinking_prefill_retries == 0
    assert agent._last_content_with_tools is None
    assert agent._last_content_tools_all_housekeeping is False
    assert agent._mute_post_response is False


def test_in_turn_guard_defers_when_recent_real_provider_usage_proved_rough_estimate_noisy():
    agent = _agent(defer=True)
    history = _messages()

    result = _maybe_compress_before_api_call(
        agent,
        history,
        system_message="system",
        active_system_prompt="active system",
        conversation_history=history,
        approx_request_tokens=150,
        effective_task_id="task-1",
        compression_attempts=0,
        max_compression_attempts=3,
        api_call_count=2,
    )

    assert result["compressed"] is False
    assert result["messages"] is history
    assert result["compression_attempts"] == 0


def test_in_turn_guard_respects_disabled_auto_compression():
    agent = _agent(compression_enabled=False)
    history = _messages()

    result = _maybe_compress_before_api_call(
        agent,
        history,
        system_message="system",
        active_system_prompt="active system",
        conversation_history=history,
        approx_request_tokens=150,
        effective_task_id="task-1",
        compression_attempts=0,
        max_compression_attempts=3,
        api_call_count=1,
    )

    assert result["compressed"] is False
    assert result["exhausted"] is False
    assert result["messages"] is history


def test_in_turn_guard_reports_exhaustion_without_compressing():
    agent = _agent()
    history = _messages()

    result = _maybe_compress_before_api_call(
        agent,
        history,
        system_message="system",
        active_system_prompt="active system",
        conversation_history=history,
        approx_request_tokens=150,
        effective_task_id="task-1",
        compression_attempts=3,
        max_compression_attempts=3,
        api_call_count=9,
    )

    assert result["compressed"] is False
    assert result["exhausted"] is True
    assert result["messages"] is history
