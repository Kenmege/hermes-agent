"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt_parts


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _captured_context_cwd(agent):
    """The cwd build_system_prompt_parts hands to build_context_files_prompt."""
    captured = {}

    def fake_context_files(cwd=None, skip_soul=False, context_length=None):
        captured["cwd"] = cwd
        return ""

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", side_effect=fake_context_files),
    ):
        build_system_prompt_parts(agent)
    return captured["cwd"]


class TestContextFileCwd:
    def test_none_when_terminal_cwd_unset(self, monkeypatch):
        # Unset → None, so discovery falls back to the launch dir inside
        # build_context_files_prompt (the local-CLI #19242 contract).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert _captured_context_cwd(_make_agent()) is None

    def test_configured_dir_when_terminal_cwd_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _captured_context_cwd(_make_agent()) == tmp_path


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


class TestMemoryRailGuidance:
    def test_context_mode_guidance_injected_when_ctx_tools_loaded(self):
        stable = _stable_prompt(_make_agent(valid_tool_names=["mcp_context_mode_ctx_execute", "mcp_context_mode_ctx_search"]))
        assert "Context-mode as the context-window management rail" in stable
        assert "mcp_context_mode_ctx_batch_execute" in stable
        assert "Do not use context-mode as a substitute for required verification" in stable

    def test_byterover_guidance_injected_when_brv_tools_loaded(self):
        stable = _stable_prompt(_make_agent(valid_tool_names=["brv_query", "brv_curate"]))
        assert "ByteRover as the implementation-memory rail" in stable
        assert "Linear remains execution state" in stable
        assert "Obsidian remains human-readable" in stable

    def test_linear_guidance_uses_actual_mcp_tool_names(self):
        stable = _stable_prompt(_make_agent(valid_tool_names=["mcp_linear_get_issue", "mcp_linear_save_issue", "mcp_linear_save_comment"]))
        assert "Linear as the execution-state rail" in stable
        assert "mcp_linear_save_issue" in stable
        assert "mcp_linear_save_comment" in stable
        assert "mcp_linear_update_issue" not in stable
        assert "mcp_linear_create_comment" not in stable

    def test_obsidian_guidance_keeps_default_as_orchestrator_not_spoofed_writer(self):
        stable = _stable_prompt(_make_agent(valid_tool_names=["mcp_empire_registry_obsidian_read", "mcp_empire_registry_obsidian_write"]))
        assert "main/default Hermes profile remains" in stable
        assert "do not spoof another caller" in stable.lower()
        assert "caller` is your real profile/role id" in stable


class TestCodingContextBlock:
    def test_injected_when_active(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        stable = _stable_prompt(agent)
        assert "coding agent" in stable
        assert "Workspace" in stable

    def test_absent_when_off(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        # Drive the real path: force the resolved mode to "off" via config.
        with patch("agent.coding_context._coding_mode", return_value="off"):
            stable = _stable_prompt(agent)
        assert "coding agent" not in stable

    def test_absent_without_tools(self, monkeypatch, tmp_path):
        import subprocess

        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=[], platform="cli")
        assert "coding agent" not in _stable_prompt(agent)
