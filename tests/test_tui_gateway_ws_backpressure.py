"""Backpressure / delivery-guarantee semantics of the queued WSTransport.

Covers the 2026-06-09 root cause (session 20260609_173231_25a56c): a
slow-but-alive Desktop client used to wedge pool-thread writes for 10s,
permanently mark the transport closed, and drop the final visible render of a
completed turn. The queued transport must instead (a) never block writers,
(b) shed only droppable status/delta frames under pressure, and (c) keep
responses and message.complete deliverable.
"""

import asyncio

from tui_gateway import ws as ws_mod


def _frame(event_type: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "event",
        "params": {"type": event_type, "session_id": "s"},
    }


def test_droppable_classification():
    assert ws_mod._is_droppable_frame(_frame("message.delta"))
    assert ws_mod._is_droppable_frame(_frame("status.update"))
    assert ws_mod._is_droppable_frame(_frame("thinking.delta"))
    # Never droppable: the final render, errors, approvals, responses.
    assert not ws_mod._is_droppable_frame(_frame("message.complete"))
    assert not ws_mod._is_droppable_frame(_frame("message.start"))
    assert not ws_mod._is_droppable_frame(_frame("error"))
    assert not ws_mod._is_droppable_frame(_frame("approval.request"))
    assert not ws_mod._is_droppable_frame({"jsonrpc": "2.0", "id": 7, "result": {}})
    assert not ws_mod._is_droppable_frame("not-a-dict")


def test_write_never_blocks_and_sheds_only_droppable_frames(monkeypatch):
    """A wedged client must not block writers; deltas shed, completes kept."""
    monkeypatch.setattr(ws_mod, "_WS_QUEUE_SOFT_MAX", 8)
    monkeypatch.setattr(ws_mod, "_WS_QUEUE_HARD_MAX", 64)

    async def scenario():
        release = asyncio.Event()
        sent: list[str] = []

        class WedgedWS:
            async def send_text(self, line):
                await release.wait()
                sent.append(line)

        transport = ws_mod.WSTransport(WedgedWS(), asyncio.get_running_loop(), peer="t")

        # Saturate past the soft cap with droppable deltas (writer is wedged
        # on the first frame, so the queue only drains after release).
        for _ in range(40):
            assert transport.write(_frame("message.delta"))

        # The essential final render still enqueues above the soft cap.
        assert transport.write(_frame("message.complete"))
        await asyncio.sleep(0)

        assert transport.dropped_frames > 0
        # Queue holds at most soft-max droppables (+ the in-flight frame
        # already popped by the writer) + the essential frame.
        assert len(transport._queue) <= ws_mod._WS_QUEUE_SOFT_MAX + 1

        release.set()
        for _ in range(200):
            if not transport._queue:
                break
            await asyncio.sleep(0.01)

        assert any('"message.complete"' in line for line in sent)
        transport.close()
        await asyncio.sleep(0)

    asyncio.run(scenario())


def test_hard_overflow_declares_transport_dead(monkeypatch):
    monkeypatch.setattr(ws_mod, "_WS_QUEUE_SOFT_MAX", 4)
    monkeypatch.setattr(ws_mod, "_WS_QUEUE_HARD_MAX", 8)

    async def scenario():
        class WedgedWS:
            async def send_text(self, line):
                await asyncio.Event().wait()  # never completes

        transport = ws_mod.WSTransport(WedgedWS(), asyncio.get_running_loop(), peer="t")

        # Non-droppable frames pile up past the hard cap.
        results = [
            transport.write({"jsonrpc": "2.0", "id": i, "result": {}}) for i in range(20)
        ]
        await asyncio.sleep(0)

        assert results[-1] is False or transport._closed
        assert transport.write(_frame("message.complete")) is False
        transport.close()
        await asyncio.sleep(0)

    asyncio.run(scenario())


def test_write_async_confirms_delivery_and_fails_closed():
    async def scenario():
        sent: list[str] = []

        class OkWS:
            async def send_text(self, line):
                sent.append(line)

        loop = asyncio.get_running_loop()
        transport = ws_mod.WSTransport(OkWS(), loop, peer="t")
        ok = await transport.write_async({"jsonrpc": "2.0", "id": 1, "result": {}})
        assert ok is True
        assert len(sent) == 1

        transport.close()
        await asyncio.sleep(0)
        assert await transport.write_async({"jsonrpc": "2.0", "id": 2, "result": {}}) is False

    asyncio.run(scenario())


def test_thread_writes_marshal_onto_loop():
    """Pool-thread write() must enqueue via call_soon_threadsafe and return
    immediately (the old path blocked up to 10s and could mark the transport
    dead on a slow client)."""

    async def scenario():
        sent: list[str] = []
        done = asyncio.Event()

        class OkWS:
            async def send_text(self, line):
                sent.append(line)
                if '"message.complete"' in line:
                    done.set()

        loop = asyncio.get_running_loop()
        transport = ws_mod.WSTransport(OkWS(), loop, peer="t")

        def writer_thread():
            return transport.write(_frame("message.complete"))

        result = await asyncio.to_thread(writer_thread)
        assert result is True
        await asyncio.wait_for(done.wait(), 5)
        transport.close()
        await asyncio.sleep(0)

    asyncio.run(scenario())
