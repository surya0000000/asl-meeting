"""Unit tests for temporal sequence buffering."""

import numpy as np
import pytest

from ml.sequence_buffer import SequenceBuffer


def test_sequence_buffer_add_and_shape() -> None:
    buffer = SequenceBuffer(sequence_length=3, feature_dim=4)
    buffer.add([1, 2, 3, 4])
    buffer.add(np.array([5, 6, 7, 8], dtype=np.float32))

    assert buffer.size == 2
    arr = buffer.as_array()
    assert arr.shape == (2, 4)
    assert not buffer.is_full


def test_sequence_buffer_sliding_window_replaces_oldest() -> None:
    buffer = SequenceBuffer(sequence_length=2, feature_dim=3)
    buffer.add([1, 1, 1])
    buffer.add([2, 2, 2])
    buffer.add([3, 3, 3])

    assert buffer.is_full
    arr = buffer.as_array()
    np.testing.assert_allclose(arr[0], np.array([2, 2, 2], dtype=np.float32))
    np.testing.assert_allclose(arr[1], np.array([3, 3, 3], dtype=np.float32))


def test_sequence_buffer_validates_feature_dim() -> None:
    buffer = SequenceBuffer(sequence_length=2, feature_dim=3)
    with pytest.raises(ValueError):
        buffer.add([1, 2])

