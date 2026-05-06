from rag_guardrails import find_uncited_claims, split_claims, validate_answer_citations


def test_split_claims_splits_sentence_like_units():
    assert split_claims("Alpha is grounded [S1]. Beta needs support.") == [
        "Alpha is grounded [S1].",
        "Beta needs support.",
    ]


def test_find_uncited_claims_ignores_short_fragments():
    uncited = find_uncited_claims(
        "Yes. Retrieval augmented generation combines search and generation. Grounded claim [S1]."
    )

    assert uncited == ["Retrieval augmented generation combines search and generation."]


def test_validate_answer_citations_flags_unknown_labels():
    report = validate_answer_citations("Grounded answer [S2].", {"S1"})

    assert report["ok"] is False
    assert report["unknown_labels"] == ["S2"]
