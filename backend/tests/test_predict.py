"""REST predict endpoint tests."""

from __future__ import annotations


def test_predict_with_sequence(client) -> None:
    sequence = [[0.02] * 63, [0.03] * 63]
    response = client.post(
        "/predict",
        json={"sequence": sequence, "refine_text": False, "speak": False},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "raw_prediction" in payload
    assert "confidence" in payload
    assert payload["refined_text"] != ""


def test_predict_with_single_frame_buffering(client) -> None:
    response = client.post(
        "/predict",
        json={"landmarks": [0.05] * 63, "refine_text": False, "speak": False},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["raw_prediction"] in {"BUFFERING", "HELLO", "YES", "NO", "WAIT", "QUESTION", "AGREE", "DISAGREE", "SLOW DOWN", "THANK YOU", "UNSURE"}

