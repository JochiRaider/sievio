from dataclasses import dataclass

from sievio.core.config import build_config_from_defaults_and_options


@dataclass
class _SampleCfg:
    a: int = 0
    b: int = 0


def test_defaults_only_populates_config():
    cfg = build_config_from_defaults_and_options(
        _SampleCfg,
        defaults={"a": 1, "b": 2},
        options=None,
    )
    assert cfg == _SampleCfg(a=1, b=2)


def test_options_override_defaults():
    cfg = build_config_from_defaults_and_options(
        _SampleCfg,
        defaults={"a": 1, "b": 2},
        options={"b": 3},
    )
    assert cfg == _SampleCfg(a=1, b=3)


def test_unknown_and_ignored_keys_are_dropped():
    cfg = build_config_from_defaults_and_options(
        _SampleCfg,
        defaults={"a": 1},
        options={"a": 2, "unknown": 3},
        ignore_keys=("a",),
    )
    assert cfg == _SampleCfg(a=1, b=0)
