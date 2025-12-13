# main.py
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from ..core.config import SievioConfig, load_config_from_path
from ..core.dataset_card import build_dataset_card_from_fragments
from ..core.log import configure_logging
from ..core.qc_post import _run_post_qc
from ..core.sharding import SHARDING_STRATEGIES, generate_shard_configs
from ..core.stats_aggregate import merge_pipeline_stats
from .runner import (
    convert,
    convert_local_dir,
    convert_github,
    make_local_repo_config,
    make_github_repo_config,
)

try:  # pragma: no cover - optional dependency
    from ..core.extras.qc import JSONLQualityScorer, score_jsonl_to_csv
except Exception:  # pragma: no cover
    JSONLQualityScorer = None  # type: ignore[assignment]
    score_jsonl_to_csv = None  # type: ignore[assignment]


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level Sievio CLI argument parser.

    Configures the main parser with subcommands for running pipelines,
    converting local directories or GitHub repositories, building dataset
    cards, and running post-hoc QC.

    Returns:
        argparse.ArgumentParser: Configured argument parser for the CLI.
    """
    parser = argparse.ArgumentParser(prog="sievio", description="Sievio CLI")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., DEBUG, INFO, WARNING).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run from a config file")
    run_p.add_argument("-c", "--config", required=True, help="Path to config file (TOML or JSON).")
    run_p.add_argument("--override-max-workers", type=int, help="Override pipeline.max_workers.")
    run_p.add_argument(
        "--override-executor-kind",
        choices=["auto", "thread", "process"],
        help="Override pipeline.executor_kind.",
    )
    run_p.add_argument("--dry-run", action="store_true", help="Validate and print config, then exit.")

    local_p = subparsers.add_parser("local", help="Run local directory -> JSONL.")
    local_p.add_argument("root_dir", help="Root directory to process.")
    local_p.add_argument("out_jsonl", help="Output JSONL path.")
    local_p.add_argument("--prompt", help="Optional prompt output path.")
    local_p.add_argument("--base-config", help="Optional base config TOML/JSON.")

    gh_p = subparsers.add_parser("github", help="Run GitHub URL -> JSONL.")
    gh_p.add_argument("url", help="GitHub repository URL.")
    gh_p.add_argument("out_jsonl", help="Output JSONL path.")
    gh_p.add_argument("--prompt", help="Optional prompt output path.")
    gh_p.add_argument("--base-config", help="Optional base config TOML/JSON.")

    card_p = subparsers.add_parser("card", help="Build dataset card from fragments.")
    card_p.add_argument(
        "--fragments",
        nargs="+",
        required=True,
        help='Glob(s) for *.card.json fragments (e.g., "out/*.card.json").',
    )
    card_p.add_argument("--output", required=True, help="Path to write rendered README.md.")

    qc_p = subparsers.add_parser("qc", help="Run post-hoc QC on an existing JSONL.")
    qc_p.add_argument("input", help="Input JSONL path.")
    qc_p.add_argument("--csv", help="Optional CSV output path.")
    qc_p.add_argument("--parallel", action="store_true", help="Enable parallel QC if extras allow.")
    qc_p.add_argument("--config", help="Optional config to reuse QC settings.")

    shard_p = subparsers.add_parser("shard", help="Generate sharded configs.")
    shard_p.add_argument("--targets", required=True, type=Path, help="File of targets.")
    shard_p.add_argument("--base", required=True, type=Path, help="Base config TOML/JSON.")
    shard_p.add_argument("--shards", required=True, type=int, help="Number of shards.")
    shard_p.add_argument("--out-dir", required=True, type=Path, help="Output directory for shard configs.")
    shard_p.add_argument(
        "--kind",
        required=True,
        choices=sorted(SHARDING_STRATEGIES),
        help="Source kind to shard.",
    )

    merge_p = subparsers.add_parser("merge-stats", help="Merge stats JSON files.")
    merge_p.add_argument("stats_files", nargs="+", type=Path, help="Paths to stats JSON files.")
    merge_p.add_argument("--output", "-o", type=Path, help="Output file (defaults to stdout).")

    return parser


def _apply_pipeline_overrides(cfg: SievioConfig, args: argparse.Namespace) -> None:
    """Apply pipeline-related CLI overrides to a config object.

    Updates the provided ``SievioConfig`` in place using any
    override flags supplied on the command line, such as maximum worker
    count or executor kind.

    Args:
        cfg (SievioConfig): Configuration object to mutate.
        args (argparse.Namespace): Parsed CLI arguments that may contain
            pipeline override fields.
    """
    if getattr(args, "override_max_workers", None) is not None:
        cfg.pipeline.max_workers = int(args.override_max_workers)
    if getattr(args, "override_executor_kind", None):
        cfg.pipeline.executor_kind = args.override_executor_kind


def _load_base_config(path: Optional[str]) -> Optional[SievioConfig]:
    """Load an optional base configuration from a file path.

    When ``path`` is provided, the configuration is loaded using
    ``load_config_from_path``; otherwise, ``None`` is returned.

    Args:
        path (str | None): Path to a TOML or JSON configuration file, or
            None to skip loading.

    Returns:
        SievioConfig | None: Loaded configuration instance, or None
        if no path was provided.
    """
    if not path:
        return None
    return load_config_from_path(path)


def _cmd_shard(args: argparse.Namespace) -> int:
    """Generate sharded configs from a base config and targets list."""
    cfg = load_config_from_path(args.base)
    raw_lines = args.targets.read_text("utf-8").splitlines()
    targets = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise SystemExit(f"Refusing to write shard configs into non-empty {out_dir}")

    for shard_id, shard_cfg in generate_shard_configs(
        input_list=targets,
        base_config=cfg,
        num_shards=args.shards,
        source_kind=args.kind,
    ):
        out_path = out_dir / f"shard_{shard_id}.json"
        out_path.write_text(json.dumps(shard_cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
    return 0


def _cmd_merge_stats(args: argparse.Namespace) -> int:
    """Merge stats JSON files and write to stdout or a file."""
    stats_dicts = []
    for path in args.stats_files:
        data = json.loads(path.read_text("utf-8"))
        stats_dicts.append(data)

    merged = merge_pipeline_stats(stats_dicts)
    text = json.dumps(merged, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    """Dispatch a parsed CLI command to the appropriate handler.

    Executes the selected subcommand (``run``, ``local``, ``github``,
    ``card``, or ``qc``), wiring configuration, running conversions or
    QC, and printing results or errors to standard output.

    Args:
        args (argparse.Namespace): Parsed arguments from the top-level
            argument parser.

    Returns:
        int: Process exit code, where 0 indicates success and non-zero
        values indicate failure.
    """
    configure_logging(level=args.log_level)
    cmd = args.command

    if cmd == "run":
        cfg = load_config_from_path(args.config)
        _apply_pipeline_overrides(cfg, args)
        if args.dry_run:
            cfg.validate()
            print(json.dumps(cfg.to_dict(), indent=2))
            return 0
        stats = convert(cfg)
        print(json.dumps(stats, indent=2))
        return 0

    if cmd == "local":
        base_cfg = _load_base_config(args.base_config)
        stats = convert_local_dir(
            root_dir=args.root_dir,
            out_jsonl=args.out_jsonl,
            out_prompt=args.prompt,
            base_config=base_cfg,
        )
        print(json.dumps(stats, indent=2))
        return 0

    if cmd == "github":
        base_cfg = _load_base_config(args.base_config)
        stats = convert_github(
            url=args.url,
            out_jsonl=args.out_jsonl,
            out_prompt=args.prompt,
            base_config=base_cfg,
        )
        print(json.dumps(stats, indent=2))
        return 0

    if cmd == "card":
        fragment_paths: list[Path] = []
        for pattern in args.fragments:
            fragment_paths.extend(Path(p) for p in glob.glob(pattern))
        if not fragment_paths:
            print("No card fragments found.", file=sys.stderr)
            return 1
        rendered = build_dataset_card_from_fragments(fragment_paths)
        Path(args.output).write_text(rendered, encoding="utf-8")
        return 0

    if cmd == "qc":
        jsonl_path = Path(args.input)
        if not jsonl_path.exists():
            print(f"Input JSONL not found: {jsonl_path}", file=sys.stderr)
            return 1
        cfg = _load_base_config(args.config) or SievioConfig()
        cfg.metadata.primary_jsonl = str(jsonl_path)
        cfg.qc.enabled = True
        cfg.qc.mode = "post"
        cfg.qc.parallel_post = bool(args.parallel)
        cfg.qc.write_csv = False  # handled separately for explicit --csv
        try:
            summary = _run_post_qc(str(jsonl_path), cfg)
        except Exception as exc:  # noqa: BLE001
            msg = "QC extras are required for this command." if JSONLQualityScorer is None else str(exc)
            print(msg, file=sys.stderr)
            return 1
        if args.csv:
            if score_jsonl_to_csv is None:
                print("CSV output requested but QC extras are not installed.", file=sys.stderr)
                return 1
            score_jsonl_to_csv(str(jsonl_path), str(args.csv))
        print(json.dumps(summary, indent=2))
        return 0

    if cmd == "shard":
        return _cmd_shard(args)

    if cmd == "merge-stats":
        return _cmd_merge_stats(args)

    print(f"Unknown command: {cmd}", file=sys.stderr)
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the Sievio command-line interface.

    Parses arguments, dispatches to the selected subcommand, and returns
    an appropriate process exit code. This function is used by the
    console script entry point as well as direct ``python -m`` invocations.

    Args:
        argv (Sequence[str] | None): Optional list of argument strings to
            parse instead of ``sys.argv[1:]``. Primarily useful for tests.

    Returns:
        int: Process exit code, where 0 indicates success and non-zero
        values indicate failure.
    """
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return _dispatch(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
