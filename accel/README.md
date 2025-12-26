# sievio-accel

**sievio-accel** is an optional Rust extension module for [Sievio](../README.md). It provides high-performance, drop-in replacements for CPU-intensive hot paths in the ingestion pipeline, currently focusing on Quality Control (QC) algorithms.

> **Note:** This package is completely optional. Sievio functions correctly without it, falling back to pure-Python implementations.

## Features

### Current Acceleration (QC)
The module currently accelerates hashing and similarity algorithms used in deduplication and contamination checks. All heavy computations **release the GIL**, allowing true parallelism in threaded pipelines.

- **SimHash (64-bit)**: Fast Locality Sensitive Hashing (LSH) for near-duplicate detection.
- **MinHash**: Deterministic MinHash signatures for estimating Jaccard similarity, compatible with Datasketch-style coefficients.

### Roadmap
Future evaluations will target other CPU-bound loops in the pipeline once strict parity testing is complete:
- Text Decoding (Mojibake repair/normalization)
- Chunking (Token-aware splitting)
- Record Conversion (Format transformation)

## Installation

### For Users
This package is usually installed as an extra with the main `sievio` package (if wheels are available).


### For Developers

To build and install the extension in editable mode (requires Rust toolchain + `maturin`):

```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Install dev dependencies
pip install maturin

# 3. Build and install into the current environment
maturin develop --manifest-path accel/Cargo.toml

```

## Usage

The extension exposes its functionality via the `sievio_accel` namespace. In the main `sievio` codebase, these are used via a "try-import" shim pattern.

```python
from sievio_accel.qc import simhash64, minhash_signature_for_text

text = "The quick brown fox jumps over the lazy dog."

# Compute SimHash (returns int)
# Releases GIL during tokenization and hashing.
fingerprint = simhash64(text)
print(f"SimHash: {fingerprint:016x}")

# Compute MinHash Signature
# Releases GIL; useful for estimating Jaccard similarity.
signature = minhash_signature_for_text(
    text,
    k=5,            # Shingle size (5-grams)
    n_perm=128,     # Number of permutations
    max_shingles=20000
)
print(f"Signature (first 5): {signature[:5]}")

```

## Architecture & Parity

* **Correctness First**: Rust implementations are strictly tested for parity with their Python equivalents. Output is deterministic and identical to the reference implementation.
* **Thread Safety**: Heavy functions detach from the Python runtime (`allow_threads`), enabling efficient execution alongside Python's I/O-bound threads.
* **Safe Fallbacks**: The main library does not import this module directly; it uses an adapter that checks for presence, ensuring `sievio` runs anywhere Python runs.

See [`LLMS.md`](LLMS.md) and [`AGENTS.md`](AGENTS.md) in this directory for detailed architectural rules and contribution guidelines.