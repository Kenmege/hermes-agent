"""Tool result persistence -- preserves large outputs instead of truncating.

Defense against context-window overflow operates at three levels:

1. **Per-tool output cap** (inside each tool): Tools like search_files
   pre-truncate their own output before returning. This is the first line
   of defense and the only one the tool author controls.

2. **Per-result persistence** (maybe_persist_tool_result): After a tool
   returns, if its output exceeds the tool's registered threshold
   (registry.get_max_result_size), the full output is written INTO THE
   SANDBOX temp dir (for example /tmp/hermes-results/{tool_use_id}.txt on
   standard Linux, or $TMPDIR/hermes-results/{tool_use_id}.txt on Termux)
   via env.execute(). The in-context content is replaced with a preview +
   file path reference. The model can read_file to access the full output
   on any backend.

3. **Per-turn aggregate budget** (enforce_turn_budget): After all tool
   results in a single assistant turn are collected, if the total exceeds
   MAX_TURN_BUDGET_CHARS (200K), the largest non-persisted results are
   spilled to disk until the aggregate is under budget. This catches cases
   where many medium-sized results combine to overflow context.
"""

import logging
import os
import shlex
import uuid

from tools.budget_config import (
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
    DEFAULT_BUDGET,
)

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
STORAGE_DIR = "/tmp/hermes-results"
HEREDOC_MARKER = "HERMES_PERSIST_EOF"
_BUDGET_TOOL_NAME = "__budget_enforcement__"


def _resolve_storage_dir(env) -> str:
    """Return the best temp-backed storage dir for this environment."""
    if env is not None:
        get_temp_dir = getattr(env, "get_temp_dir", None)
        if callable(get_temp_dir):
            try:
                temp_dir = get_temp_dir()
            except Exception as exc:
                logger.debug("Could not resolve env temp dir: %s", exc)
            else:
                if temp_dir:
                    temp_dir = temp_dir.rstrip("/") or "/"
                    return f"{temp_dir}/hermes-results"
    return STORAGE_DIR


def generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """Truncate at last newline within max_chars. Returns (preview, has_more)."""
    if len(content) <= max_chars:
        return content, False
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        truncated = truncated[:last_nl + 1]
    return truncated, True


def _heredoc_marker(content: str) -> str:
    """Return a heredoc delimiter that doesn't collide with content."""
    if HEREDOC_MARKER not in content:
        return HEREDOC_MARKER
    return f"HERMES_PERSIST_{uuid.uuid4().hex[:8]}"


def _write_to_sandbox(content: str, remote_path: str, env) -> bool:
    """Write content into the sandbox via env.execute(). Returns True on success.

    Pushes ``content`` through stdin rather than embedding it in the command
    string. Linux's ``MAX_ARG_STRLEN`` caps any single argv element at 128 KB
    (32 * PAGE_SIZE), so the previous heredoc-in-the-command-string approach
    silently failed with ``OSError: [Errno 7] Argument list too long`` for any
    tool result over ~128 KB — exactly the case persistence exists to handle.
    Routing through stdin removes that ceiling on local + ssh (``_stdin_mode
    == "pipe"``); remote backends with ``_stdin_mode == "heredoc"`` keep their
    existing API-body sized limit, which is orders of magnitude larger than
    the exec-arg ceiling.
    """
    storage_dir = os.path.dirname(remote_path)
    cmd = f"mkdir -p {shlex.quote(storage_dir)} && cat > {shlex.quote(remote_path)}"
    result = env.execute(cmd, timeout=30, stdin_data=content)
    return result.get("returncode", 1) == 0


def _build_persisted_message(
    preview: str,
    has_more: bool,
    original_size: int,
    file_path: str,
    ctx_indexed: bool = False,
) -> str:
    """Build the <persisted-output> replacement block."""
    size_kb = original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    msg += f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
    msg += f"Full output saved to: {file_path}\n"
    msg += "Use the read_file tool with offset and limit to access specific sections of this output.\n"
    if ctx_indexed:
        # Bridge: the persisted output is now also searchable via context-mode (ctx-),
        # so the model can query it by content instead of re-reading the full file.
        # This is what makes ctx- earn its keep on large tool results.
        msg += (
            "This output is also indexed for semantic search: use the `ctx_batch_execute` / "
            "context-mode `search` tool with a query to retrieve relevant sections without "
            "reading the whole file.\n"
        )
    msg += f"\nPreview (first {len(preview)} chars):\n"
    msg += preview
    if has_more:
        msg += "\n..."
    msg += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    return msg


# ── context-mode bridge ────────────────────────────────────────────────
# When a large tool result is persisted to the sandbox, ALSO index it into the
# context-mode (ctx-) FTS5 knowledge base so it becomes searchable by content.
# This is the bridge that makes ctx- actually engage on large outputs: instead of
# the model re-reading a huge file, it can `search` the indexed content. Disabled
# by default unless the context-mode binary is present; toggle off with
# HERMES_CTX_BRIDGE=0. Failure is non-fatal — tool execution never breaks if ctx-
# is unavailable.

_CTX_BINARY_CANDIDATES = (
    os.path.expanduser("~/.local/bin/context-mode"),
    "/usr/local/bin/context-mode",
)


def _resolve_ctx_binary() -> str | None:
    """Return the context-mode binary path if available, else None."""
    env_bin = os.environ.get("HERMES_CTX_BINARY")
    if env_bin and os.path.isfile(env_bin):
        return env_bin
    for candidate in _CTX_BINARY_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


def _index_to_context_mode(
    sandbox_path: str,
    tool_name: str,
    tool_use_id: str,
    env=None,
) -> bool:
    """Index a persisted tool-result file into context-mode's FTS5 store.

    Runs `context-mode index <path>` so the spilled output becomes searchable.
    Returns True if indexed, False otherwise. Never raises — ctx- is an
    enhancement layer, not a dependency.
    """
    if os.environ.get("HERMES_CTX_BRIDGE", "1") == "0":
        return False
    # Only index local files (sandbox writes to remote backends aren't reachable
    # by the local ctx- binary).
    if not sandbox_path.startswith("/") or not os.path.isfile(sandbox_path):
        return False
    binary = _resolve_ctx_binary()
    if binary is None:
        return False
    import subprocess
    source_label = f"hermes-persisted:{tool_name}:{tool_use_id}"
    cmd = [binary, "index", sandbox_path, "--source", source_label]
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ},
        )
        if result.returncode == 0:
            logger.info(
                "context-mode bridge: indexed persisted %s output (%s) -> ctx-",
                tool_name, tool_use_id,
            )
            return True
        logger.debug(
            "context-mode bridge: index failed for %s (rc=%s): %s",
            tool_use_id, result.returncode, result.stderr[:200],
        )
    except Exception as exc:
        logger.debug("context-mode bridge: index error for %s: %s", tool_use_id, exc)
    return False


def maybe_persist_tool_result(
    content: str,
    tool_name: str,
    tool_use_id: str,
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
    threshold: int | float | None = None,
) -> str:
    """Layer 2: persist oversized result into the sandbox, return preview + path.

    Writes via env.execute() so the file is accessible from any backend
    (local, Docker, SSH, Modal, Daytona). Falls back to inline truncation
    if write fails or no env is available.

    Args:
        content: Raw tool result string.
        tool_name: Name of the tool (used for threshold lookup).
        tool_use_id: Unique ID for this tool call (used as filename).
        env: The active BaseEnvironment instance, or None.
        config: BudgetConfig controlling thresholds and preview size.
        threshold: Explicit override; takes precedence over config resolution.

    Returns:
        Original content if small, or <persisted-output> replacement.
    """
    effective_threshold = threshold if threshold is not None else config.resolve_threshold(tool_name)

    if effective_threshold == float("inf"):
        return content

    if len(content) <= effective_threshold:
        return content

    storage_dir = _resolve_storage_dir(env)
    remote_path = f"{storage_dir}/{tool_use_id}.txt"
    preview, has_more = generate_preview(content, max_chars=config.preview_size)

    if env is not None:
        sandbox_ok = False
        try:
            sandbox_ok = _write_to_sandbox(content, remote_path, env)
        except Exception as exc:
            logger.warning("Sandbox write failed for %s: %s", tool_use_id, exc)

        if sandbox_ok:
            logger.info(
                "Persisted large tool result: %s (%s, %d chars -> %s)",
                tool_name, tool_use_id, len(content), remote_path,
            )
            # Bridge: also index the persisted file into context-mode so it is
            # searchable by content (not just a file the model must read_file).
            # Deliberately OUTSIDE the sandbox-write try block: a bridge failure
            # (even an unexpected one) must NEVER cause truncation-after-success,
            # because the sandbox write already succeeded. The helper swallows
            # its own errors, but this placement is belt-and-braces.
            try:
                ctx_indexed = _index_to_context_mode(remote_path, tool_name, tool_use_id, env)
            except Exception as exc:
                logger.debug("context-mode bridge error for %s: %s", tool_use_id, exc)
                ctx_indexed = False
            return _build_persisted_message(
                preview, has_more, len(content), remote_path, ctx_indexed=ctx_indexed
            )

    logger.info(
        "Inline-truncating large tool result: %s (%d chars, no sandbox write)",
        tool_name, len(content),
    )
    return (
        f"{preview}\n\n"
        f"[Truncated: tool response was {len(content):,} chars. "
        f"Full output could not be saved to sandbox.]"
    )


def enforce_turn_budget(
    tool_messages: list[dict],
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
) -> list[dict]:
    """Layer 3: enforce aggregate budget across all tool results in a turn.

    If total chars exceed budget, persist the largest non-persisted results
    first (via sandbox write) until under budget. Already-persisted results
    are skipped.

    Mutates the list in-place and returns it.
    """
    candidates = []
    total_size = 0
    for i, msg in enumerate(tool_messages):
        content = msg.get("content", "")
        size = len(content)
        total_size += size
        if PERSISTED_OUTPUT_TAG not in content:
            candidates.append((i, size))

    if total_size <= config.turn_budget:
        return tool_messages

    candidates.sort(key=lambda x: x[1], reverse=True)

    for idx, size in candidates:
        if total_size <= config.turn_budget:
            break
        msg = tool_messages[idx]
        content = msg["content"]
        tool_use_id = msg.get("tool_call_id", f"budget_{idx}")

        replacement = maybe_persist_tool_result(
            content=content,
            tool_name=_BUDGET_TOOL_NAME,
            tool_use_id=tool_use_id,
            env=env,
            config=config,
            threshold=0,
        )
        if replacement != content:
            total_size -= size
            total_size += len(replacement)
            tool_messages[idx]["content"] = replacement
            logger.info(
                "Budget enforcement: persisted tool result %s (%d chars)",
                tool_use_id, size,
            )

    return tool_messages
