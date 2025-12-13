import json
from pathlib import Path

from sievio.cli.main import main
from sievio.core.config import SievioConfig, SinkSpec, SourceSpec
from sievio.core.interfaces import RepoContext


def _make_config_file(tmp_path: Path) -> tuple[Path, Path, Path]:
    cfg = SievioConfig()
    ctx = RepoContext(repo_full_name="local/test", repo_url="https://example.com/local", license_id="UNKNOWN")
    cfg.sinks.context = ctx

    src_root = tmp_path / "input"
    src_root.mkdir()
    (src_root / "file.txt").write_text("hello world", encoding="utf-8")
    cfg.sources.specs = (SourceSpec(kind="local_dir", options={"root_dir": str(src_root)}),)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jsonl_path = out_dir / "data.jsonl"
    prompt_path = out_dir / "data.prompt.txt"

    cfg.sinks.specs = (
        SinkSpec(
            kind="default_jsonl_prompt",
            options={
                "jsonl_path": str(jsonl_path),
                "prompt_path": str(prompt_path),
            },
        ),
    )
    config_path = out_dir / "config.json"
    cfg.to_json(config_path)
    return config_path, jsonl_path, prompt_path


def test_cli_run_from_config(tmp_path: Path, capsys):
    config_path, jsonl_path, prompt_path = _make_config_file(tmp_path)

    rc = main(["run", "-c", str(config_path)])
    assert rc == 0
    out = capsys.readouterr().out
    stats = json.loads(out)
    assert stats["records"] >= 1
    assert jsonl_path.exists()
    # dataset card sidecar should be written by default
    sidecar = Path(f"{jsonl_path}.card.json")
    assert sidecar.exists()
    assert prompt_path.exists()


def test_cli_local_command(tmp_path: Path, capsys):
    root = tmp_path / "input"
    root.mkdir()
    (root / "file.txt").write_text("hi", encoding="utf-8")
    out_jsonl = tmp_path / "out.jsonl"
    prompt = tmp_path / "out.prompt.txt"

    rc = main(["local", str(root), str(out_jsonl), "--prompt", str(prompt)])
    assert rc == 0
    out = capsys.readouterr().out
    stats = json.loads(out)
    assert stats["records"] >= 1
    assert out_jsonl.exists()
    assert prompt.exists()


def test_cli_card_build(tmp_path: Path):
    frag1 = {
        "schema_version": 1,
        "file": "a.jsonl",
        "split": "train",
        "num_examples": 2,
        "num_bytes": 10,
        "language": ["en"],
    }
    frag2 = {
        "schema_version": 1,
        "file": "b.jsonl",
        "split": "test",
        "num_examples": 1,
        "num_bytes": 5,
        "language": ["en"],
    }
    f1 = tmp_path / "frag1.card.json"
    f2 = tmp_path / "frag2.card.json"
    f1.write_text(json.dumps(frag1), encoding="utf-8")
    f2.write_text(json.dumps(frag2), encoding="utf-8")
    output = tmp_path / "README.md"

    rc = main(["card", "--fragments", str(f1), str(f2), "--output", str(output)])
    assert rc == 0
    text = output.read_text(encoding="utf-8")
    assert "---" in text  # YAML header present
    assert "language:" in text
