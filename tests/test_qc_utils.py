import hashlib
import random
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from sievio.core import qc_utils
from sievio.core.qc_utils import (
    MinHashLSH,
    minhash_signature_for_text,
    parse_ok,
    repetition_rate,
    simhash64,
)


def test_parse_ok_rejects_partial_json():
    assert parse_ok('{"a": 1', "json") == 0.0
    assert parse_ok('{"a": 1}\n{invalid', "jsonl") == 0.0


def test_parse_ok_handles_invalid_yaml():
    try:
        import yaml  # noqa: F401
    except Exception:
        pytest.skip("yaml not available")

    assert parse_ok("key: : value\n -", "yaml") == 0.0


def test_parse_ok_handles_malformed_restructuredtext_or_markdown():
    # Very short or heading-free text should fall back to a reduced confidence score.
    assert parse_ok("nonsense content", "markdown") < 1.0
    assert parse_ok("title\n---\nbody", "restructuredtext") == pytest.approx(1.0)


def test_simhash64_single_token_matches_hash():
    text = "Token"
    expected = int.from_bytes(
        hashlib.blake2b(text.lower().encode("utf-8"), digest_size=8).digest(),
        "little",
    )
    assert simhash64(text) == expected


def test_simhash64_non_positive_max_tokens_returns_zero():
    text = "Token"
    assert simhash64(text, max_tokens=0) == 0
    assert simhash64(text, max_tokens=-5) == 0


def test_open_jsonl_output_maybe_gz_normalizes_mode_and_rejects_binary(tmp_path):
    path = tmp_path / "out.jsonl.gz"
    with qc_utils.open_jsonl_output_maybe_gz(path, "a") as fp:
        fp.write("line\n")
    with qc_utils.open_jsonl_maybe_gz(path) as fp:
        assert fp.read() == "line\n"
    with pytest.raises(ValueError):
        qc_utils.open_jsonl_output_maybe_gz(path, "ab")


def test_minhash_lsh_rejects_invalid_params_and_signatures():
    with pytest.raises(ValueError):
        MinHashLSH(n_perm=0)
    with pytest.raises(ValueError):
        MinHashLSH(n_perm=10, bands=3)
    with pytest.raises(ValueError):
        MinHashLSH(n_perm=10, bands=5, jaccard_threshold=1.5)

    lsh = MinHashLSH(n_perm=8, bands=2)
    with pytest.raises(ValueError):
        lsh.add_and_check("doc", (1, 2, 3))


def test_repetition_rate_caps_grams_deterministically():
    text = "abcd " * 50
    k = 4
    max_grams = 10
    truncated = text[: max_grams + k - 1]
    assert repetition_rate(text, k=k, max_grams=max_grams) == pytest.approx(
        repetition_rate(truncated, k=k, max_grams=None)
    )


def test_repetition_rate_tracks_repeat_positions():
    text = "aaaaa"
    assert repetition_rate(text, k=2) == pytest.approx(0.75)


def test_minhash_signature_supports_large_perm_counts_deterministically():
    text = "abcdefg " * 200
    sig = minhash_signature_for_text(text, k=5, n_perm=256)
    assert len(sig) == 256
    assert any(val != 0xFFFFFFFF for val in sig[128:])
    assert sig == minhash_signature_for_text(text, k=5, n_perm=256)
    assert sig[:128] == minhash_signature_for_text(text, k=5, n_perm=128)


def test_minhash_signature_caps_shingles_consistently():
    text = "abcd " * 200
    k = 4
    max_shingles = 20
    truncated = text[: max_shingles + k - 1]
    assert minhash_signature_for_text(
        text,
        k=k,
        n_perm=32,
        max_shingles=max_shingles,
    ) == minhash_signature_for_text(truncated, k=k, n_perm=32, max_shingles=None)


def test_minhash_signature_treats_non_positive_max_shingles_as_none():
    text = "abcd " * 200
    k = 4
    n_perm = 32
    assert minhash_signature_for_text(
        text,
        k=k,
        n_perm=n_perm,
        max_shingles=-1,
    ) == minhash_signature_for_text(text, k=k, n_perm=n_perm, max_shingles=None)


def test_minhash_signature_is_deterministic_under_threads(monkeypatch):
    text = "abcdefg " * 200
    k = 5
    n_perm = 256

    monkeypatch.setattr(qc_utils, "_MINHASH_COEFS", [])
    monkeypatch.setattr(qc_utils, "_MINHASH_RNG", random.Random(qc_utils._MINHASH_SEED))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: minhash_signature_for_text(text, k=k, n_perm=n_perm),
                range(16),
            )
        )

    monkeypatch.setattr(qc_utils, "_MINHASH_COEFS", [])
    monkeypatch.setattr(qc_utils, "_MINHASH_RNG", random.Random(qc_utils._MINHASH_SEED))
    expected = minhash_signature_for_text(text, k=k, n_perm=n_perm)
    assert all(sig == expected for sig in results)


def test_parse_ok_respects_max_bytes():
    text = '{"a": 1}\n' * 100
    assert parse_ok(text, "jsonl", max_bytes=10) == 0.0


def test_minhash_rejects_excessive_perm_counts():
    text = "abcdefg " * 10
    with pytest.raises(ValueError):
        minhash_signature_for_text(text, k=5, n_perm=qc_utils._MINHASH_MAX_PERMS + 1)
    with pytest.raises(ValueError):
        MinHashLSH(n_perm=qc_utils._MINHASH_MAX_PERMS + 1, bands=1)


def test_perplexity_model_fail_soft_when_loading_fails(monkeypatch):
    class BoomTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise OSError("missing model")

    monkeypatch.setattr(qc_utils, "_ensure_hf", lambda: True)
    monkeypatch.setattr(qc_utils, "AutoTokenizer", BoomTokenizer)
    model = qc_utils.PerplexityModel("missing", local_files_only=True)
    assert model.model is None
    assert model.tok is None
    assert model.ppl("hello") == float("inf")


def test_perplexity_overflow_returns_inf(monkeypatch):
    class DummyTok:
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3, 4, 5]

    class DummyLoss:
        def detach(self):
            return self

        def __float__(self):
            return 1000.0

    class DummyModel:
        device = "cpu"

        def __call__(self, input_ids, labels=None):
            return SimpleNamespace(loss=DummyLoss())

    class DummyTensor:
        def __init__(self, data, device="cpu"):
            self._data = [list(data[0])]
            self.device = device

        def size(self, dim):
            return len(self._data[0]) if dim == 1 else len(self._data)

        def clone(self):
            return DummyTensor([list(self._data[0])], device=self.device)

        def __getitem__(self, idx):
            _, col_slice = idx
            cols = self._data[0][col_slice]
            return DummyTensor([cols], device=self.device)

        def __setitem__(self, idx, value):
            _, col_slice = idx
            indices = range(len(self._data[0]))[col_slice]
            for i in indices:
                self._data[0][i] = value

    class DummyNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyTorch:
        @staticmethod
        def tensor(data, device=None):
            return DummyTensor(data, device=device or "cpu")

        @staticmethod
        def no_grad():
            return DummyNoGrad()

    monkeypatch.setattr(qc_utils, "torch", DummyTorch())
    model = qc_utils.PerplexityModel.__new__(qc_utils.PerplexityModel)
    model.model = DummyModel()
    model.tok = DummyTok()
    model.max_len = 4
    model.stride = 2
    assert model.ppl("hello") == float("inf")
