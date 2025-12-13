from sievio.core.dataset_card import CardFragment, _aggregate_signal_stats


def test_aggregate_signal_stats_qc_summary_shape():
    frag = CardFragment(
        file="test.jsonl",
        num_examples=1,
        extra={
            "stats": {
                "qc_summary": {
                    "signal_stats": {
                        "foo": {"count": 2, "mean": 1.0, "min": 1.0, "max": 1.0, "stdev": 0.0},
                    }
                }
            }
        },
    )

    aggregated = _aggregate_signal_stats([frag])

    assert aggregated is not None
    assert aggregated["foo"]["count"] == 2
    assert aggregated["foo"]["mean"] == 1.0


def test_aggregate_signal_stats_pipeline_shape():
    frag = CardFragment(
        file="test.jsonl",
        num_examples=1,
        extra={
            "stats": {
                "qc": {
                    "signal_stats": {
                        "foo": {"count": 2, "mean": 1.0, "min": 1.0, "max": 1.0, "stdev": 0.0},
                    }
                }
            }
        },
    )

    aggregated = _aggregate_signal_stats([frag])

    assert aggregated is not None
    assert aggregated["foo"]["mean"] == 1.0
    assert "quality:foo" not in aggregated


def test_aggregate_signal_stats_falls_back_to_quality_screener():
    frag = CardFragment(
        file="test.jsonl",
        num_examples=1,
        extra={
            "stats": {
                "qc": {
                    "signal_stats": {},
                    "screeners": {
                        "quality": {
                            "signal_stats": {
                                "q_sig": {"count": 1, "mean": 2.0, "min": 2.0, "max": 2.0, "stdev": 0.0}
                            }
                        },
                        "safety": {"signal_stats": {"s_sig": {"count": 1, "mean": 3.0, "min": 3.0, "max": 3.0, "stdev": 0.0}}},
                    },
                }
            }
        },
    )

    aggregated = _aggregate_signal_stats([frag])

    assert aggregated is not None
    assert aggregated["q_sig"]["mean"] == 2.0
    assert "safety:s_sig" not in aggregated
