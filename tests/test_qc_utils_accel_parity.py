from concurrent.futures import ThreadPoolExecutor

import pytest

from sievio.core import accel as accel_utils
from sievio.core import qc_utils

pytest.importorskip("sievio_accel")


def test_simhash64_parity_with_accel(monkeypatch):
    text = "Token"
    accel_utils._reset_accel_cache_for_tests()
    monkeypatch.setenv("SIEVIO_ACCEL", "0")
    python_val = qc_utils.simhash64(text)

    accel_utils._reset_accel_cache_for_tests()
    monkeypatch.setenv("SIEVIO_ACCEL", "1")
    accel_val = qc_utils.simhash64(text)
    assert accel_val == python_val


def test_minhash_parity_with_accel(monkeypatch):
    text = "abcdefg " * 200
    accel_utils._reset_accel_cache_for_tests()
    monkeypatch.setenv("SIEVIO_ACCEL", "0")
    python_sig = qc_utils.minhash_signature_for_text(text, k=5, n_perm=128)

    accel_utils._reset_accel_cache_for_tests()
    monkeypatch.setenv("SIEVIO_ACCEL", "1")
    accel_sig = qc_utils.minhash_signature_for_text(text, k=5, n_perm=128)
    assert accel_sig == python_sig


def test_simhash_non_positive_max_tokens_matches_python(monkeypatch):
    text = "Token"
    accel_utils._reset_accel_cache_for_tests()
    monkeypatch.setenv("SIEVIO_ACCEL", "0")
    python_val = qc_utils.simhash64(text, max_tokens=-5)

    accel_utils._reset_accel_cache_for_tests()
    monkeypatch.setenv("SIEVIO_ACCEL", "1")
    accel_val = qc_utils.simhash64(text, max_tokens=-5)
    assert accel_val == python_val


def test_minhash_non_positive_max_shingles_matches_python(monkeypatch):
    text = "abcd " * 200
    k = 4
    n_perm = 32
    accel_utils._reset_accel_cache_for_tests()
    monkeypatch.setenv("SIEVIO_ACCEL", "0")
    python_sig = qc_utils.minhash_signature_for_text(
        text,
        k=k,
        n_perm=n_perm,
        max_shingles=-1,
    )

    accel_utils._reset_accel_cache_for_tests()
    monkeypatch.setenv("SIEVIO_ACCEL", "1")
    accel_sig = qc_utils.minhash_signature_for_text(
        text,
        k=k,
        n_perm=n_perm,
        max_shingles=-1,
    )
    assert accel_sig == python_sig


def test_accel_minhash_deterministic_under_threads(monkeypatch):
    text = "abcdefg " * 200
    k = 5
    n_perm = 128
    accel_utils._reset_accel_cache_for_tests()
    monkeypatch.setenv("SIEVIO_ACCEL", "1")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: qc_utils.minhash_signature_for_text(text, k=k, n_perm=n_perm),
                range(16),
            )
        )

    expected = qc_utils.minhash_signature_for_text(text, k=k, n_perm=n_perm)
    assert all(sig == expected for sig in results)
