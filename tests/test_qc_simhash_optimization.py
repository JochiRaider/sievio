import pytest

from sievio.core.qc_utils import SimHashWindowIndex, hamming


def test_simhash_index_basic():
    idx = SimHashWindowIndex(window_size=5, hamming_thresh=4)

    h1 = 0
    idx.add(h1, "doc1")

    dist, doc_id = idx.query(h1)
    assert dist == 0
    assert doc_id == "doc1"

    h2 = 1  # distance 1 from h1
    dist, doc_id = idx.query(h2)
    assert dist == 1
    assert doc_id == "doc1"

    h_far = 0x0001000100010001  # no band overlap with h1
    dist, doc_id = idx.query(h_far)
    assert dist is None
    assert doc_id is None


def test_simhash_window_eviction():
    idx = SimHashWindowIndex(window_size=2, hamming_thresh=4)

    idx.add(0, "A")
    idx.add(1, "B")

    dist, _ = idx.query(0)
    assert dist == 0

    idx.add(0xFFFFFFFFFFFFFFFF, "C")

    dist, doc_id = idx.query(0)
    assert dist == 1
    assert doc_id == "B"
