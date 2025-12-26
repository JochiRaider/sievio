# LLMS.md (accel/) - architecture and invariants for Rust + PyO3 acceleration

This directory inherits the repository-root `LLMS.md` and `AGENTS.md`.

This file is intentionally short: it defines accel-specific invariants for correctness,
parallelism, and packaging. Implementation workflow and checklists live in `accel/AGENTS.md`.

---

## 1) Non-negotiables

- The accel package is optional. Core Python must run and pass tests with no Rust toolchain
  and no compiled wheels installed.
- Python semantics are authoritative. Rust must match Python behavior (including edge cases
  and determinism) unless the task explicitly scopes a change and adds tests.
- Do not introduce new required runtime dependencies for core `sievio`.
- Python core resolves accel availability via the central shim `src/sievio/core/accel.py` (do not import sievio_accel directly elsewhere).

---

## 2) Responsibility split

Keep in Python:
- I/O and filesystem walking
- Network access
- Pipeline orchestration, routing, registries, and sinks/sources
- Configuration parsing and policy decisions

Move to Rust (candidates):
- CPU-heavy, allocation-heavy, pure(ish) transforms
- Hashing, shingling, similarity signatures, token-ish heuristics
- Batch transforms over many inputs

---

## 3) PyO3 boundary rules (attach/detach contract)

Acceleration boundary:
- Attached conversion -> Detached compute -> Attached wrapping

Terminology note:
- Modern PyO3 uses `Python::attach` / `Python::detach` to reflect free-threaded Python support.
  Older PyO3 used `Python::with_gil` / `Python::allow_threads`. Code should use the pinned version;
  policy text uses attach/detach terminology.
- The pinned PyO3 version in accel/Cargo.toml is authoritative for which API names are used.

Rules:
- CPU-bound Rust must detach for the compute phase using the pinned PyO3 API.
- While detached:
  - Do not touch Python objects (including `PyAny`, `PyList`, `Bound<'py, T>`, or `Py<T>`)
  - Do not call into Python or create Python objects
  - Do not construct or raise `PyErr`
  - Capture only Rust-owned data (`Vec<u8>`, `String`, domain structs)
  - Capture errors in Rust-owned values and convert to `PyErr` only after re-attach
- While attached:
  - Use the Bound API (`Bound<'py, T>`) for object interaction

---

## 4) Threading policy (default sequential; parallel only when amortized)

- Scalar functions default to sequential execution.
- Batch functions may use internal parallelism (e.g., Rayon) when enabled by the thread budget.
- Avoid nested parallelism:
  - If the Python pipeline is already parallel (thread pool or process pool), Rust internal threads
    default to 1 unless explicitly justified and measured for a single-kernel win.

Thread budget (single source of truth):
- Primary knob: `SIEVIO_ACCEL_THREADS` (default 1).
- If Rayon is used, `RAYON_NUM_THREADS` may be honored only as a fallback when
  `SIEVIO_ACCEL_THREADS` is not set.
- Do not spawn unbounded threads.
- Thread budget is read at initialization; changing it requires a process restart.

---

## 5) Determinism requirements

- Outputs must not depend on thread scheduling order.
- Avoid parallel floating-point reductions unless you implement a stable reduction strategy.
  Prefer integer accumulators or explicitly ordered aggregation.

---

## 6) Safety and deadlocks

- Do not hold Rust locks across Python interaction or attach/detach transitions.
- Avoid global mutable state unless required and tested under concurrency.
- Do not rely on a GIL for correctness; Rust concurrency must be safe on its own.

---

## 7) Packaging guidance

- Keep the Python-facing surface small and stable (functions over exposed Rust types).
- Prefer abi3 wheels when feasible for the supported CPython floor:
  - If abi3 is used, set the minimum version feature (e.g., `abi3-py311`) and treat abi3
    API constraints as part of the contract.
  - If abi3 is not feasible, document why and keep the surface minimal.

---

## 8) Definition of done for accel PRs

- Python tests pass with no accel installed.
- Same tests pass with accel installed and the accelerated path is exercised.
- Parity tests exist for representative edge cases.
- If internal parallelism exists, determinism is tested across thread counts.
- Benchmarks demonstrate improvement for the intended workload and record thread settings used.

<!-- END accel/LLMS.md -->