from sievio.core.extras.qc import JSONLQualityScorer


def test_global_deduplication_scores_and_flags(tmp_path) -> None:
    db_path = tmp_path / "global_dedup.db"
    scorer = JSONLQualityScorer(
        enable_minhash=False,
        minhash_perms=32,
        minhash_bands=8,
        minhash_shingle_k=3,
        minhash_jaccard_thresh=0.5,
        global_dedup_path=str(db_path),
    )

    rec_a = {"text": "hello world", "meta": {"doc_id": "a"}}
    rec_b = {"text": "hello world", "meta": {"doc_id": "b"}}

    res_a = scorer.score_record(rec_a)
    res_b = scorer.score_record(rec_b)

    assert res_b["global_dup"] is True
    assert res_b["near_dup"] is True
    assert res_b["dup_family_id"] == "a"
    assert res_b["global_dup_of"] == "a"
    assert res_b["score"] < res_a["score"]
