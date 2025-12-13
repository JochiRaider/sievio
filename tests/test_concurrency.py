import threading
import time

import pytest

from sievio.core.config import SievioConfig
from sievio.core.concurrency import Executor, ExecutorConfig, infer_executor_kind, process_items_parallel


def _process_one_echo(item):
    return (item, [item])


def test_map_unordered_submit_error_handled_and_continues():
    cfg = ExecutorConfig(max_workers=2, window=2, kind="thread")
    executor = Executor(cfg)

    items = [1, 2, 3]
    seen = []
    submit_errors = []

    def fn(x):
        return x * 2

    def on_result(res):
        seen.append(res)

    def on_submit_error(item, exc):
        submit_errors.append((item, exc))

    # Monkeypatch submit to fail for item == 2
    original_make = executor._make_executor

    class SubmitFailingExecutor:
        def __init__(self, real):
            self.pool = real

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.pool.__exit__(exc_type, exc, tb)  # type: ignore[attr-defined]

        def submit(self, fn_, item):
            if item == 2:
                raise RuntimeError("submit fail")
            return self.pool.submit(fn_, item)

    executor._make_executor = lambda: SubmitFailingExecutor(original_make())  # type: ignore[assignment]

    executor.map_unordered(
        items,
        fn,
        on_result,
        fail_fast=False,
        on_submit_error=on_submit_error,
    )

    assert set(seen) == {2, 6}  # 2 was skipped due to submit failure
    assert len(submit_errors) == 1
    assert submit_errors[0][0] == 2


def test_process_items_parallel_delegates_submit_and_worker_errors():
    items = [1, 2, 3]
    results = []
    submit_errors = []
    worker_errors = []

    def process_one(x):
        if x == 3:
            raise ValueError("boom")
        return (x, [x])

    def write_records(item, recs):
        results.append((item, list(recs)))

    def on_submit_error(item, exc):
        submit_errors.append(item)

    def on_worker_error(exc):
        worker_errors.append(str(exc))

    process_items_parallel(
        items,
        process_one,
        write_records,
        max_workers=2,
        window=2,
        fail_fast=False,
        on_submit_error=on_submit_error,
        on_worker_error=on_worker_error,
    )

    assert results == [(1, [1]), (2, [2])]
    assert worker_errors == ["boom"]
    assert submit_errors == []


def test_process_items_parallel_process_executor_pickling_on_submit_error():
    # Use a non-picklable lock to force a submission-time PicklingError in the
    # process pool path and ensure on_submit_error is invoked.
    bad_item = threading.Lock()
    submit_errors = []

    def on_submit_error(item, exc):
        submit_errors.append((item, str(exc)))

    try:
        process_items_parallel(
            [bad_item],
            _process_one_echo,
            lambda item, recs: None,  # write_records should never be called
            max_workers=2,
            window=2,
            fail_fast=False,
            on_submit_error=on_submit_error,
            executor_kind="process",
        )
    except PermissionError as exc:
        pytest.skip(f"Process pool unavailable in environment: {exc}")

    assert len(submit_errors) == 1
    bad_item_seen, msg = submit_errors[0]
    assert bad_item_seen is bad_item
    assert "pickle" in msg.lower()


def test_map_unordered_respects_window_backpressure():
    cfg = ExecutorConfig(max_workers=2, window=2, kind="thread")
    executor = Executor(cfg)

    submitted: list[tuple[int, float]] = []
    release = threading.Event()
    results: list[int] = []

    def slow_worker(x):
        release.wait()
        return x

    def on_result(res):
        results.append(res)

    original_make = executor._make_executor

    class RecordingExecutor:
        def __init__(self, real):
            self.pool = real

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.pool.__exit__(exc_type, exc, tb)  # type: ignore[attr-defined]

        def submit(self, fn_, item):
            submitted.append((item, time.monotonic()))
            return self.pool.submit(fn_, item)

    executor._make_executor = lambda: RecordingExecutor(original_make())  # type: ignore[assignment]

    t = threading.Thread(
        target=lambda: executor.map_unordered(
            [1, 2, 3, 4],
            slow_worker,
            on_result,
            fail_fast=False,
        )
    )
    t.start()
    time.sleep(0.1)
    assert len(submitted) == 2  # window=2 should block further submissions
    release.set()
    t.join(timeout=5)
    assert len(submitted) == 4
    assert set(results) == {1, 2, 3, 4}


def test_infer_executor_respects_attributes():
    def dummy_heavy_handler(data, path, ctx, pol):  # pragma: no cover - exercised via inference
        return []

    dummy_heavy_handler.cpu_intensive = True

    cfg = SievioConfig()
    cfg.pipeline.bytes_handlers = [(None, dummy_heavy_handler)]

    kind = infer_executor_kind(cfg)
    assert kind == "process"
