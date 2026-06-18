from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }



def _make_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[_tool("terminal"), _tool("mcp_linear_get_issue")]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent



def test_build_api_kwargs_uses_effective_tools_without_mutating_agent_tools():
    agent = _make_agent()
    full_tools = list(agent.tools)
    effective_tools = [_tool("terminal")]
    agent._effective_tools_for_current_request = effective_tools

    kwargs = agent._build_api_kwargs([{"role": "user", "content": "run date"}])

    assert kwargs["tools"] == effective_tools
    assert agent.tools == full_tools
    assert [tool["function"]["name"] for tool in agent.tools] == ["terminal", "mcp_linear_get_issue"]



def test_build_api_kwargs_falls_back_to_full_tools_when_no_effective_override():
    agent = _make_agent()
    agent._effective_tools_for_current_request = None

    kwargs = agent._build_api_kwargs([{"role": "user", "content": "run date"}])

    assert kwargs["tools"] == agent.tools
