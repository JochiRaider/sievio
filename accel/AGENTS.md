# AGENTS.md (accel/) - guidance for native acceleration work (Rust + PyO3)

This directory contains the optional Rust acceleration add-on for Sievio (PyO3 + maturin).
It inherits the repository-root `AGENTS.md` rules; follow those first.

This file defines the accel operational ruleset and safety contract for changes in `accel/`.
If `accel/LLMS.md` exists, treat it as the architecture map; this file defines implementation
constraints, review requirements, and definition-of-done.

---

## 0) Primary objective

Accelerate CPU-heavy hot paths while preserving Python-visible behavior and keeping the
acceleration layer optional. Expose a small, stable Python API via PyO3 and use Rust-side
parallelism only when it produces real wall-clock wins without breaking determinism, safety,
or observability.

---

## 1) Scope

Do:
- Implement hot-path, pure(ish) accelerators (CPU-heavy, allocation-heavy helpers), starting
  with QC utilities (hashing, shingling, token-ish heuristics, similarity signatures).
- Preserve observable Python semantics (inputs, outputs, edge cases, determinism) unless the
  task explicitly scopes a behavior change and adds tests for it.
- Keep I/O, network access, and pipeline orchestration in Python. The Rust layer must not own
  sinks/sources, filesystem walking, HTTP, config loading, or record routing.

Do not:
- Reimplement pipeline control flow, retries, batching policy, logging policy, or registry wiring.
- Introduce new required runtime dependencies for core `sievio`.
- Add behavior that changes results depending on thread count, platform, CPU features, or locale.

---

## 2) Package & boundary goals (maximum decoupling)

Optional install contract:
- Core `sievio` must run with no Rust toolchain and no compiled wheels installed.
- Import failures must gracefully fall back to the pure-Python implementation.

Public API contract:
- Expose a small, stable Python-facing API. Prefer functions mirroring existing Python helpers
  rather than exporting Rust types.
- Keep the public surface domain-oriented (e.g. `sievio_accel.qc`) so new accelerators can
  be added later without changing the import pattern.

Compatibility posture:
- Treat Python behavior as the reference implementation. Rust must match it.
- Prefer abi3 wheels when feasible, but never sacrifice correctness for ABI stability.

---

## 3) Layout guidance (one distribution, one extension module)

Preferred approach:
- One accelerator distribution (e.g., `sievio-accel`) that ships one compiled extension module
  (`sievio_accel._native`), with multiple "domains" exposed as Python submodules.

Implementation shape:
- `sievio_accel/qc.py` re-exports `sievio_accel._native.qc_*` functions.
- Do not create separate compiled modules per domain unless strictly necessary.

---

## 4) Import shim pattern (core stays authoritative)

In core Python modules (e.g., `src/sievio/core/accel.py`):
- Core Python should import accel only in the shim module; all other modules call through the shim.
- Import accelerated functions behind a single local indirection layer.
- Otherwise use the existing Python implementation.
- Do not scatter `try/except ImportError` throughout the file.

---

## 5) Rust/PyO3 engineering rules (FFI safety and stability)

FFI safety:
- Do not intentionally panic from Rust called by Python.
- Return `PyResult<T>` and convert Rust errors into Python exceptions (`PyErr`) while attached.

Modern PyO3 API posture:
- Prefer the Bound API (`Bound<'py, T>`) for object interaction while attached.
- Avoid deprecated primitives; prefer current PyO3 synchronization/caching primitives when needed.

Signature discipline:
- Prefer primitives (`&str`, `&[u8]`, `u64`, `f64`) over complex Python objects.
- If you must accept Python objects, accept `&Bound<'_, PyAny>` and convert immediately to Rust-owned data.

Memory policy:
- Avoid surprising allocations.
- Never return views into Rust-owned memory that can dangle after the call.

Unsafe policy:
- Avoid `unsafe` unless demonstrably necessary.
- Document invariants and fuzz-test any `unsafe` blocks.

---

## 6) Attach/Detach parallelism contract (the central rule set)

PyO3 terminology note:
- In newer PyO3, `Python::attach` / `Python::detach` are the primary names. Older versions used
  `Python::with_gil` / `Python::allow_threads`. Use the pinned version in code, but keep this
  policy described in attach/detach terms.

The goal:
- Convert Python inputs to Rust-owned data while attached.
- Detach thread state for CPU-bound Rust compute.
- Re-attach to wrap results and raise exceptions.

Core rules:
1) Separate conversion from compute:
   - Phase A (attached): validate inputs, extract bytes/primitives, allocate Rust-owned data.
   - Phase B (detached): compute using Rust-only data. No Python objects, no Python calls.
   - Phase C (attached): wrap results into Python objects and raise exceptions if needed.

2) While detached:
   - Do not touch Python objects (including `PyAny`, `PyList`, `Bound<'py, T>`, or `Py<T>`).
   - Do not create or raise `PyErr`.
   - Capture only Rust-owned values (`Vec<u8>`, `String`, domain structs) and translate errors on re-attach.

3) Sequential scalar vs parallel batch:
   - Scalar APIs default to sequential.
   - Batch APIs may use Rayon when enabled by the thread budget and when overhead is amortized.
   - Default to 1 thread: batch APIs remain sequential unless `SIEVIO_ACCEL_THREADS > 1`.

4) Unified control (one knob wins):
   - Primary knob: `SIEVIO_ACCEL_THREADS` (default 1).
   - Define ‘initialization’ as Rust module initialization (extension import) unless the implementation explicitly documents otherwise.
   - Honor `RAYON_NUM_THREADS` only as a fallback when `SIEVIO_ACCEL_THREADS` is not set.
   - Reading the thread budget at initialization implies changes require a process restart.

5) Avoid nested parallelism traps:
   - If the Python pipeline is already parallel (`max_workers > 1`), Rust internal parallelism defaults
     to 1 unless explicitly justified and measured for a single-kernel win.

6) Verify correctness under different thread counts:
   - Parity tests must pass with `SIEVIO_ACCEL_THREADS=1` and `SIEVIO_ACCEL_THREADS=4`.

---

## 7) Build & test workflow (venv-first)

Note: `maturin` lives in the project venv.

1) Activate the repo venv (`source .venv/bin/activate`).
2) Build/install into the venv: `python -m maturin develop`
   - run from accel/ (where Cargo.toml lives)
4) From the repo root, run the repo-root test command (canonical; keep in sync with `../AGENTS.md`).
   - Current repo-root command: `pytest`
5) Run Rust tests inside `accel/`: `cargo test`

---

## 8) Testing standards (parity and edge cases)

Parity testing is mandatory:
- Inputs: empty strings, Unicode (emojis / normalization), large inputs, pathological repetition.
- Behavior: match Python reference semantics exactly (including "soft caps").
- Determinism: identical results across runs and thread counts.

---

## 9) Performance validation

PRs claiming speedups must include a benchmark measuring:
1) Python reference.
2) Rust accel (sequential).
3) Rust accel (parallel, if applicable), including thread settings used.

---

## 10) Packaging & distribution invariants

- Keep the Python-facing surface small and stable (functions over exposed Rust types).
- Prefer abi3 wheels when feasible:
  - If abi3 is used, set the minimum supported CPython version via the appropriate PyO3 feature
    (e.g., `abi3-py311`) and treat abi3 constraints as part of the contract.
  - If abi3 is not feasible (required CPython APIs exceed abi3), document why.

---

## 11) Review guidelines (P0 / P1)

P0 (Required):
- Parity with Python reference.
- Correct attach/detach usage (no Python objects while detached).
- No mandatory dependency on accel.

P1 (Risk):
- New accelerated domains.
- Changes to thread defaults or knob priority.
- New module-level caches or globals.

---

## 12) Definition of done for an accelerator PR

- Core passes tests with no accel installed.
- Core passes tests with accel installed and the accelerated path is exercised.
- Run the repo-root required checks (tests/lint/typecheck) per root AGENTS.md.
- At least one test explicitly targets the accelerator.
- Parity verified for representative edge cases.
- If internal parallelism exists, determinism is tested across thread counts.
- Benchmarks included for performance claims.

---

## 13) Implementation checklist for a new accelerator function

1) Identify the Python reference function and write down the exact contract:
   - Inputs, outputs, error behavior, edge cases, determinism expectations.

2) Implement the Rust function with:
   - Simple signature (primitives preferred).
   - Clear split between attached conversion and detached compute.
   - No Python callbacks inside compute.
   - Convert errors to `PyErr` only after re-attach.

3) If internal parallelism is justified, add a batch API:
   - Keep scalar API sequential.
   - Batch API uses bounded Rayon threads controlled by `SIEVIO_ACCEL_THREADS`.

4) Expose via the single compiled module (`sievio_accel._native`) and add a thin Python wrapper.

5) Wire the core import shim to prefer accel when available.

6) Add parity tests against the Python reference.

7) Validate concurrency:
   - Safe under multiple caller threads.
   - No unbounded internal parallelism.

8) Run baseline and accelerated test suites.

9) Only then consider micro-optimizations.
