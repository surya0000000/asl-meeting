"""WebSocket endpoint tests."""

from __future__ import annotations


def test_websocket_stream_prediction(client) -> None:
    frame = [0.01] * 63
    with client.websocket_connect("/ws/asl-stream") as ws:
        connected = ws.receive_json()
        assert connected["type"] == "connection"
        assert connected["status"] == "connected"

        prediction = None
        for _ in range(4):
            ws.send_json({"type": "landmarks", "landmarks": frame, "refine_text": False, "speak": False})
            message = ws.receive_json()
            if message.get("type") == "prediction":
                prediction = message
                break

        assert prediction is not None
        assert "raw_prediction" in prediction
        assert "confidence" in prediction

