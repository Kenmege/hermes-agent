"""WebSocket transport for the tui_gateway JSON-RPC server.

Reuses :func:`tui_gateway.server.dispatch` verbatim so every RPC method, every
slash command, every approval/clarify/sudo flow, and every agent event flows
through the same handlers whether the client is Ink over stdio or an iOS /
web client over WebSocket.

Wire protocol
-------------
Identical to stdio: newline-delimited JSON-RPC in both directions. The server
emits a ``gateway.ready`` event immediately after connection accept, then
echoes responses/events for inbound requests. No framing differences.

Mounting
--------
    from fastapi import WebSocket
    from tui_gateway.ws import handle_ws

    @app.websocket("/api/ws")
    async def ws(ws: WebSocket):
        await handle_ws(ws)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections import deque
import json
import logging
import socket
from typing import Any

from tui_gateway import server

_log = logging.getLogger(__name__)

# Queue limits for slow-but-alive WebSocket clients. Writers must never block on
# token/status streaming; under pressure we shed only droppable progress frames
# while preserving final responses, errors, approvals, and JSON-RPC replies.
_WS_QUEUE_SOFT_MAX = 256
_WS_QUEUE_HARD_MAX = 1024
_WS_LOG_PAYLOAD_PREVIEW = 240

# Keep starlette optional at import time; handle_ws uses the real class when
# it's available and falls back to a generic Exception sentinel otherwise.
try:
    from starlette.websockets import WebSocketDisconnect as _WebSocketDisconnect
except ImportError:  # pragma: no cover - starlette is a required install path
    _WebSocketDisconnect = Exception  # type: ignore[assignment]


def _is_droppable_frame(obj: Any) -> bool:
    """Return True for high-volume progress frames safe to shed under pressure."""
    if not isinstance(obj, dict):
        return False
    # JSON-RPC replies carry an id and must always be delivered.
    if "id" in obj:
        return False
    if obj.get("method") != "event":
        return False
    params = obj.get("params")
    if not isinstance(params, dict):
        return False
    event_type = str(params.get("type") or "")
    return event_type in {
        "message.delta",
        "thinking.delta",
        "status.update",
    }


class WSTransport:
    """Per-connection WS transport with non-blocking queued writes.

    Pool-thread writes must not block on a slow GUI/Desktop client. Frames are
    marshalled onto the owning event loop and drained by one writer task. When a
    client is slow-but-alive we shed only droppable progress frames beyond the
    soft cap; essential frames are preserved until the hard cap, where the
    transport is declared dead to avoid unbounded memory growth.
    """

    def __init__(
        self,
        ws: Any,
        loop: asyncio.AbstractEventLoop,
        *,
        peer: str = "unknown",
    ) -> None:
        self._ws = ws
        self._loop = loop
        self._peer = peer
        self._closed = False
        self._queue = deque()
        self._writer_task: asyncio.Task | None = None
        self.dropped_frames = 0

    def write(self, obj: dict) -> bool:
        if self._closed:
            return False
        line = json.dumps(obj, ensure_ascii=False)
        try:
            on_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_loop = False

        if on_loop:
            return self._enqueue(line, obj, None)

        # Cross-thread path: schedule the enqueue and return immediately.  The
        # old implementation blocked worker threads waiting on send_text(); that
        # is exactly what wedged final renders when Desktop was slow (#2026-06-09).
        try:
            self._loop.call_soon_threadsafe(self._enqueue, line, obj, None)
            return not self._closed
        except RuntimeError as exc:
            self._closed = True
            _log.warning(
                "ws enqueue failed peer=%s error_type=%s error=%s",
                self._peer, type(exc).__name__, exc,
            )
            return False

    async def write_async(self, obj: dict) -> bool:
        """Send from the owning event loop and confirm delivery."""
        if self._closed:
            return False
        line = json.dumps(obj, ensure_ascii=False)
        fut = self._loop.create_future()
        if not self._enqueue(line, obj, fut):
            if not fut.done():
                fut.set_result(False)
            return False
        return bool(await fut)

    def _enqueue(self, line: str, obj: dict, fut: asyncio.Future | None) -> bool:
        if self._closed:
            if fut is not None and not fut.done():
                fut.set_result(False)
            return False

        if _is_droppable_frame(obj) and len(self._queue) >= _WS_QUEUE_SOFT_MAX:
            self.dropped_frames += 1
            if fut is not None and not fut.done():
                fut.set_result(True)
            return True

        if len(self._queue) >= _WS_QUEUE_HARD_MAX:
            self._closed = True
            if fut is not None and not fut.done():
                fut.set_result(False)
            return False

        self._queue.append((line, fut))
        self._ensure_writer()
        return True

    def _ensure_writer(self) -> None:
        if self._closed:
            return
        if self._writer_task is None or self._writer_task.done():
            self._writer_task = self._loop.create_task(self._drain_queue())

    async def _drain_queue(self) -> None:
        while self._queue and not self._closed:
            line, fut = self._queue.popleft()
            ok = await self._safe_send(line)
            if fut is not None and not fut.done():
                fut.set_result(ok)
            if not ok:
                break

    async def _safe_send(self, line: str) -> bool:
        try:
            await self._ws.send_text(line)
            return True
        except Exception as exc:
            self._closed = True
            _log.warning(
                "ws send failed peer=%s error_type=%s error=%s",
                self._peer, type(exc).__name__, exc,
            )
            return False

    def close(self) -> None:
        self._closed = True
        while self._queue:
            _, fut = self._queue.popleft()
            if fut is not None and not fut.done():
                fut.set_result(False)


def _ws_peer_label(ws: Any) -> str:
    """Return ``host:port`` when available, else a stable placeholder."""
    client = getattr(ws, "client", None)
    if client is None:
        return "unknown"
    host = getattr(client, "host", None) or "unknown"
    port = getattr(client, "port", None)
    return f"{host}:{port}" if port is not None else host


def _disable_nagle(ws: Any) -> None:
    """Disable Nagle so streamed JSON-RPC frames go out individually.

    Without it the kernel coalesces the small per-token frames, so a burst after
    the model's think-pause lands on the client in one tick and no client-side
    smoothing can recover the cadence. GUI/WS only; chat platforms don't hit
    this path. Best-effort — skip silently if the socket isn't reachable.
    """
    try:
        scope = getattr(ws, "scope", None) or {}
        transport = (scope.get("extensions") or {}).get("transport") or getattr(ws, "transport", None)
        sock = transport.get_extra_info("socket") if transport is not None else None
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as exc:  # pragma: no cover - best-effort tuning
        _log.debug("ws TCP_NODELAY skip: %s", exc)


async def handle_ws(ws: Any) -> None:
    """Run one WebSocket session. Wire-compatible with ``tui_gateway.entry``."""
    peer = _ws_peer_label(ws)
    transport: WSTransport | None = None
    messages = 0
    parse_errors = 0
    dispatch_crashes = 0
    send_failures = 0
    disconnect_reason = "not_connected"

    try:
        await ws.accept()
        disconnect_reason = "connected"
        # Push small streamed frames out immediately instead of letting Nagle
        # batch them — keeps the live token cadence intact for GUI clients.
        _disable_nagle(ws)
        _log.info("ws accepted peer=%s", peer)

        transport = WSTransport(ws, asyncio.get_running_loop(), peer=peer)

        ready_ok = await transport.write_async(
            {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {
                    "type": "gateway.ready",
                    "payload": {"skin": server.resolve_skin()},
                },
            }
        )
        if not ready_ok:
            disconnect_reason = "ready_send_failed"
            send_failures += 1
            _log.error("ws ready frame send failed peer=%s", peer)
            return

        while True:
            try:
                raw = await ws.receive_text()
            except _WebSocketDisconnect as exc:
                disconnect_reason = (
                    "client_disconnect("
                    f"code={getattr(exc, 'code', None)},"
                    f"reason={getattr(exc, 'reason', None)})"
                )
                break
            except Exception:
                disconnect_reason = "receive_failed"
                _log.exception("ws receive failed peer=%s", peer)
                break

            line = raw.strip()
            if not line:
                continue
            messages += 1

            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors += 1
                _log.warning(
                    "ws parse error peer=%s index=%d error=%s payload=%r",
                    peer,
                    messages,
                    exc,
                    line[:_WS_LOG_PAYLOAD_PREVIEW],
                )
                ok = await transport.write_async(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": "parse error"},
                        "id": None,
                    }
                )
                if not ok:
                    disconnect_reason = "send_failed_after_parse_error"
                    send_failures += 1
                    _log.warning("ws parse-error reply send failed peer=%s", peer)
                    break
                continue

            # dispatch() may schedule long handlers on the pool; it returns
            # None in that case and the worker writes the response itself via
            # the transport we pass in (a separate thread, so transport.write
            # is the safe path there). For inline handlers it returns the
            # response dict, which we write here from the loop.
            req_id = req.get("id") if isinstance(req, dict) else None
            req_method = req.get("method") if isinstance(req, dict) else None
            try:
                resp = await asyncio.to_thread(server.dispatch, req, transport)
            except Exception:
                dispatch_crashes += 1
                _log.exception(
                    "ws dispatch crash peer=%s id=%s method=%s",
                    peer,
                    req_id,
                    req_method,
                )
                ok = await transport.write_async(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32603, "message": "internal error"},
                        "id": req_id if req_id is not None else None,
                    }
                )
                if not ok:
                    disconnect_reason = "send_failed_after_dispatch_crash"
                    send_failures += 1
                    _log.warning(
                        "ws dispatch-crash reply send failed peer=%s id=%s method=%s",
                        peer,
                        req_id,
                        req_method,
                    )
                    break
                continue
            if resp is not None and not await transport.write_async(resp):
                disconnect_reason = "send_failed_after_response"
                send_failures += 1
                _log.warning(
                    "ws response send failed peer=%s id=%s method=%s",
                    peer,
                    req_id,
                    req_method,
                )
                break
    finally:
        reaped_sessions = 0
        detached_sessions = 0
        if transport is not None:
            transport.close()

            # Reap sessions this transport owned (close_on_disconnect sidecar
            # sessions) or detach the rest to the drop sentinel so later emits
            # don't crash into a closed socket or fall through to desktop stdout
            # logs. Detached sessions are handed to the grace-windowed WS-orphan
            # reaper inside _close_sessions_for_transport (a quick reconnect /
            # session.resume cancels it). This is the single WS-disconnect
            # teardown path.
            #
            # Offloaded: _close_session_by_id does a blocking worker.close()
            # (terminate + waits) plus a synchronous DB write — inline that
            # would freeze the uvicorn event loop for every other live
            # connection.
            try:
                reaped_sessions, detached_sessions = await asyncio.to_thread(
                    server._close_sessions_for_transport,
                    transport,
                    end_reason="ws_disconnect",
                )
            except Exception:
                _log.exception("ws transport teardown failed peer=%s", peer)
        try:
            await ws.close()
        except Exception as exc:
            _log.debug("ws close failed peer=%s error=%s", peer, exc)
        _log.info(
            "ws closed peer=%s reason=%s messages=%d parse_errors=%d "
            "dispatch_crashes=%d send_failures=%d reaped_sessions=%d detached_sessions=%d",
            peer,
            disconnect_reason,
            messages,
            parse_errors,
            dispatch_crashes,
            send_failures,
            reaped_sessions,
            detached_sessions,
        )
