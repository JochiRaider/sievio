<div id="top"></div>

<!-- PROJECT SHIELDS -->
<!--
  Reference-style links for badges live at the bottom of this file.
  This keeps the markdown a bit easier to read.
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/JochiRaider/RepoCapsule">
    <img src="images/logo.png" alt="RepoCapsule logo" width="80" height="80">
  </a>

  <h3 align="center">RepoCapsule</h3>

  <p align="center">
    Stdlib-first pipeline to turn code, docs, PDFs, and logs into clean JSONL chunks.
    <br />
    <a href="https://github.com/JochiRaider/RepoCapsule"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/JochiRaider/RepoCapsule/issues">Report Bug</a>
    ·
    <a href="https://github.com/JochiRaider/RepoCapsule/issues">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://github.com/JochiRaider/RepoCapsule)

RepoCapsule is a **stdlib-first ingestion pipeline** that turns:

- Local or GitHub-hosted repositories (code + docs)
- Web-hosted PDFs
- Windows EVTX event logs

into a **normalized JSONL dataset** that is ready for:

- LLM fine-tuning / pre-training
- Retrieval-augmented generation (RAG)
- Search and analytics pipelines

### Why RepoCapsule?

There are many one-off scripts to scrape a repo or split a PDF. RepoCapsule focuses on being:

- **Safe by default** – zip-bomb defenses, size caps, safe HTTP client, and license detection.
- **Format aware** – Markdown / reStructuredText aware chunking, code vs doc heuristics, and KQL block extraction.
- **Composable** – small, testable building blocks (`Source`, `Sink`, `ChunkPolicy`, etc.) wired together via a `RepocapsuleConfig` and `run_pipeline` / `convert` entrypoints.
- **Stdlib-first** – relies mainly on the Python standard library, with *optional* extras for tokenization, PDF parsing, EVTX, and quality scoring.

At a high level, RepoCapsule:

1. **Discovers files** (respecting `.gitignore` and skip lists).
2. **Safely decodes bytes** into text.
3. **Splits text into semantic blocks** (headings, paragraphs, code fences, sections).
4. **Chunks blocks into model-sized windows** using `ChunkPolicy` (with optional overlap).
5. **Attaches rich metadata** (paths, languages, license info, duplication families, QC scores).
6. **Streams JSONL records** (and optional prompt text) to your chosen sinks.

<p align="right">(<a href="#top">back to top</a>)</p>


### Built With

RepoCapsule is primarily a pure-Python project.

- [Python](https://www.python.org/) (3.11+)
- Python standard library (e.g., `pathlib`, `zipfile`, `concurrent.futures`, `urllib`)
- Optional: [tiktoken](https://github.com/openai/tiktoken) for exact token counting
- Optional: a PDF backend (see project extras / `pyproject.toml`)
- Optional: an EVTX parser for Windows event logs
- Optional: QC / scoring extras for dataset quality reports

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This section shows how to get a local copy of RepoCapsule up and running, either as a library or from a cloned repo.

### Prerequisites

You will need:

- **Python 3.11+**
- **pip** (and optionally `venv` or another virtual environment manager)

On many systems you can verify this with:

```sh
python --version
pip --version
```

### Installation

#### Option 1: Install from source (recommended while iterating on the project)

1. **Clone the repo**

   ```sh
   git clone https://github.com/JochiRaider/RepoCapsule.git
   cd RepoCapsule
   ```

2. **Create & activate a virtual environment (optional but recommended)**

   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install RepoCapsule in editable mode**

   ```sh
   pip install -e .
   ```

   For optional features (exact token counting, PDFs, EVTX logs), use the extras defined in `pyproject.toml`:

   ```sh
   pip install -e ".[tok]"   # tiktoken-based exact token counts
   pip install -e ".[pdf]"   # pypdf-based PDF parsing
   pip install -e ".[evtx]"  # python-evtx-based EVTX parsing
   ```

#### Option 2: Install from PyPI (if / when published)

```sh
pip install repocapsule
# or, with optional extras (see pyproject extras)
pip install "repocapsule[tok]"   # tiktoken-based exact token counts
pip install "repocapsule[pdf]"   # pypdf-based PDF parsing
pip install "repocapsule[evtx]"  # python-evtx-based EVTX parsing
```

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

RepoCapsule is designed as a library-first toolkit. The main pieces you will interact with are:

- `RepocapsuleConfig` – top-level configuration object.
- `Source` implementations – where bytes come from (local dirs, GitHub zips, web PDFs, EVTX, ...).
- `ChunkPolicy` and chunking helpers – how text gets split into model-sized chunks.
- `Sink` implementations – where normalized records go (JSONL, prompt text, custom sinks).
- `convert` / `run_pipeline` – orchestration entrypoints.

> **Note**
> The examples below are intentionally minimal and focus on the *shape* of the pipeline. For exact signatures and all options, see `config.py`, `runner.py`, `factories.py`, and `sinks.py` in this repository.


### RepocapsuleConfig at a glance

`RepocapsuleConfig` is a dataclass that groups all of the knobs for the ingestion pipeline into nested blocks (see `config.py`). At the top level it contains:

- `sources: SourceConfig`
  - `sources`: explicit `Source` objects (advanced usage; helpers like `make_local_dir_source` / `make_github_zip_source` usually populate this for you).
  - `local: LocalDirSourceConfig` – defaults for walking local directories (extensions, hidden files, `.gitignore`, per-file size caps).
  - `github: GitHubSourceConfig` – defaults for GitHub ZIPs / API downloads (per-file caps, total uncompressed cap, allowed extensions, zip-bomb protection).
  - `pdf: PdfSourceConfig` – defaults for web-PDF ingestion (timeouts, maximum PDF size, maximum links per page, whether to require PDFs, retry policy, user agent).
- `decode: DecodeConfig`
  - How raw bytes are normalised and decoded (Unicode normalisation, stripping control characters, fixing mojibake, optional per-file byte cap).
- `chunk: ChunkConfig`
  - `policy: ChunkPolicy` – the chunking strategy (e.g., `mode`, `target_tokens`, `overlap_tokens`, `min_tokens`).
  - `tokenizer_name`: optional exact-token tokenizer name used when the `tok` extra is installed.
  - `attach_language_metadata`: whether to attach language guesses to each record.
- `pipeline: PipelineConfig`
  - `extractors`: `Extractor` instances that turn files into logical text blocks (Markdown sections, code blocks, etc.).
  - `bytes_handlers`: sniff-and-decode handlers; normally left as-is so `prepare()` can populate sensible defaults.
  - `max_workers`: thread-pool size (0 = choose automatically based on CPU).
  - `submit_window`: optional cap on the number of in-flight tasks (for backpressure), or `None` for unbounded.
  - `fail_fast`: abort the run on the first error instead of best-effort processing.
- `sinks: SinkConfig`
  - `sinks`: output sinks (e.g. JSONL, prompt text, custom sinks).
  - `context`: `RepoContext` describing the source repository; attached to each record.
  - `output_dir`: where sink outputs are written.
  - `primary_jsonl_name`: name for the main JSONL file (used by QC helpers and some convenience functions).
  - `prompt: PromptConfig` – controls prompt heading format and whether a `.prompt.txt` file is emitted.
- `http: HttpConfig`
  - Shared HTTP client settings (timeout, redirect policy, allowed redirect suffixes). `prepare()` creates a `SafeHttpClient` and installs it globally.
- `qc: QCConfig`
  - Toggles dataset-quality scoring (if QC extras are installed), thresholds, duplicate detection, CSV summaries, and whether QC runs inline vs as a post-processing step.
- `logging: LoggingConfig`
  - Global logging level / format applied by `prepare()`.
- `metadata: Mapping[str, object]`
  - Free-form metadata you can attach to the run (e.g. `run_id`, `source_url`, `experiment`).

In normal usage you only override the pieces you care about, then pass the config into `convert(cfg)`. The `convert` entrypoint calls `cfg.prepare()` for you, so you do not need to call it manually unless you are wiring up the pipeline yourself.

### Converting a local repository

```python
import json
from repocapsule import RepocapsuleConfig
from repocapsule.runner import convert  # or: from repocapsule import convert

cfg = RepocapsuleConfig()

# Configure your sources/sinks.
# Typical steps (see config.py for the concrete fields):
#   - cfg.sources.sources: list of Source objects (e.g., local directories, web PDFs).
#   - cfg.sinks.sinks: output sinks (e.g., JSONL, prompt text).
#   - cfg.pipeline: concurrency and batching behaviour.
#   - cfg.qc: optional quality scoring and duplicate handling.

# After filling out cfg, run the pipeline:
stats = convert(cfg)
print(json.dumps(stats, indent=2))
```

### Converting a GitHub repo

RepoCapsule includes helpers in `githubio.py` and `runner.py` to work with GitHub repositories. A typical flow is:

1. Use `parse_github_url` to normalise any GitHub URL into a `RepoSpec`.
2. Use `default_paths_for_github` to compute output paths and a `RepoContext`.
3. Build a `RepocapsuleConfig` that includes a GitHub-based `Source`.
4. Call `convert(config)`.

### Working with PDFs

To ingest a list of web-hosted PDFs:

- Use `WebPdfListSource` with a list of direct PDF URLs.
- Or use `WebPagePdfSource` to scrape one HTML page for PDF links and then delegate to `WebPdfListSource`.

These sources enforce size caps, content sniffing, and retry/backoff logic, and they plug into the same pipeline as other sources.

### Chunking utilities

The `chunk` module is reusable on its own:

```python
from repocapsule.chunk import ChunkPolicy, count_tokens

policy = ChunkPolicy(
    mode="doc",          # or "code"
    target_tokens=1700,
    overlap_tokens=40,
    min_tokens=400,
)

text = """# My document\n\nThis is a long document you want to chunk..."""

# You can use the policy alongside the higher-level pipeline helpers
# or integrate it into your own tooling.
print("Estimated tokens:", count_tokens(text, mode="auto"))
```

### Quality & license helpers

- `licenses.detect_license_in_tree` / `detect_license_in_zip` – detect SPDX-style licenses, content licenses (e.g. Creative Commons), and attach them to a `RepoContext`.
- `export.annotate_exact_token_counts` – post-process a JSONL file to add precise token counts when the `tok` extra is installed.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

Planned and aspirational improvements include:

- [ ] Higher-level convenience builders for common configs (e.g., "just give me a GitHub URL").
- [ ] More built-in `ChunkPolicy` presets for different model sizes.
- [ ] Additional quality-scoring heuristics and visual QC reports.
- [ ] More input formats (e.g., additional log formats, archives, or markup languages).
- [ ] Example notebooks and end-to-end walkthroughs.

See the [open issues](https://github.com/JochiRaider/RepoCapsule/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See the `LICENSE` file for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Maintainer: [@JochiRaider](https://github.com/JochiRaider)

Project Link: [https://github.com/JochiRaider/RepoCapsule](https://github.com/JochiRaider/RepoCapsule)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

- [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
- [Shields.io](https://shields.io)
- All the open source projects that make Python ingestion pipelines possible.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/JochiRaider/RepoCapsule.svg?style=for-the-badge
[contributors-url]: https://github.com/JochiRaider/RepoCapsule/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/JochiRaider/RepoCapsule.svg?style=for-the-badge
[forks-url]: https://github.com/JochiRaider/RepoCapsule/network/members
[stars-shield]: https://img.shields.io/github/stars/JochiRaider/RepoCapsule.svg?style=for-the-badge
[stars-url]: https://github.com/JochiRaider/RepoCapsule/stargazers
[issues-shield]: https://img.shields.io/github/issues/JochiRaider/RepoCapsule.svg?style=for-the-badge
[issues-url]: https://github.com/JochiRaider/RepoCapsule/issues
[license-shield]: https://img.shields.io/github/license/JochiRaider/RepoCapsule.svg?style=for-the-badge
[license-url]: https://github.com/JochiRaider/RepoCapsule/blob/main/LICENSE
[product-screenshot]: images/screenshot.png

