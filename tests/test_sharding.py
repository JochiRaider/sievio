from pathlib import Path

import pytest

from sievio.core.config import SievioConfig, SinkSpec
from sievio.core.sharding import generate_shard_configs


def test_web_pdf_list_strategy():
    base = SievioConfig()
    urls = ["https://example.com/a.pdf", "https://example.com/b.pdf"]

    shards = list(generate_shard_configs(urls, base, num_shards=1, source_kind="web_pdf_list"))

    assert len(shards) == 1
    shard_id, cfg = shards[0]
    assert shard_id == "0000"
    assert len(cfg.sources.specs) == 1
    spec = cfg.sources.specs[0]
    assert spec.kind == "web_pdf_list"
    assert spec.options["urls"] == urls


def test_github_zip_scalar_strategy():
    base = SievioConfig()
    repos = ["repo1", "repo2"]

    shards = list(generate_shard_configs(repos, base, num_shards=1, source_kind="github_zip"))

    assert len(shards) == 1
    _, cfg = shards[0]
    assert len(cfg.sources.specs) == 2
    assert [spec.kind for spec in cfg.sources.specs] == ["github_zip", "github_zip"]
    assert [spec.options["url"] for spec in cfg.sources.specs] == repos


def test_isolation_of_outputs():
    base = SievioConfig()
    base.sinks.output_dir = Path("out")
    base.sinks.jsonl_basename = "data"

    shards = list(generate_shard_configs(["a", "b"], base, num_shards=2, source_kind="github_zip"))

    output_dirs = [cfg.sinks.output_dir for _, cfg in shards]
    assert output_dirs == [Path("out") / "shard_0000", Path("out") / "shard_0001"]
    basenames = [cfg.sinks.jsonl_basename for _, cfg in shards]
    assert basenames == ["data_shard_0000", "data_shard_0001"]


def test_sink_sanitization_clears_default_jsonl_prompt_paths():
    base = SievioConfig()
    base.sinks.specs = [SinkSpec(kind="default_jsonl_prompt", options={"jsonl_path": "x", "prompt_path": "y"})]

    shard_id, cfg = next(generate_shard_configs(["item"], base, num_shards=1, source_kind="github_zip"))

    assert shard_id == "0000"
    assert len(cfg.sinks.specs) == 1
    sink_spec = cfg.sinks.specs[0]
    assert sink_spec.kind == "default_jsonl_prompt"
    assert "jsonl_path" not in sink_spec.options
    assert "prompt_path" not in sink_spec.options
    # Base config should remain untouched.
    assert base.sinks.specs[0].options["jsonl_path"] == "x"
    assert base.sinks.specs[0].options["prompt_path"] == "y"


def test_unknown_source_kind_raises():
    base = SievioConfig()

    with pytest.raises(ValueError):
        list(generate_shard_configs(["input"], base, num_shards=1, source_kind="bogus"))


def test_empty_shards_skipped():
    base = SievioConfig()

    shards = list(generate_shard_configs(["a", "b"], base, num_shards=4, source_kind="github_zip"))

    assert len(shards) == 2
    assert [shard_id for shard_id, _ in shards] == ["0000", "0001"]
