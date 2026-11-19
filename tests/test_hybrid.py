"""Tests for hybrid retrieval: fusion, lexical evidence and the guard.

All pure functions, so no model, no database and no network are involved.
"""

from src import config, hybrid


def test_tokens_removes_stopwords():
    assert hybrid.tokens("What is the patent about?") == ["patent"]


def test_verbatim_statutory_phrase_is_strongest_evidence():
    text = "(k) a mathematical or business method or a computer programme per se or algorithms;"
    score = hybrid.lexical_evidence(text, ["computer programme per se"], "can I patent software")
    assert score == 1.0


def test_partial_coverage_scores_below_verbatim():
    text = "an invention which is frivolous is not patentable"
    score = hybrid.lexical_evidence(text, [], "what is not patentable in software")
    assert 0.0 < score < 1.0


def test_no_overlap_scores_zero():
    text = "the patent office shall publish the application"
    assert hybrid.lexical_evidence(text, [], "trademark fee for a small enterprise") < 0.5


def _hit(chunk_id, distance, text="t"):
    return {"chunk_id": chunk_id, "distance": distance, "text": text,
            "source": "a.pdf", "page": 1, "score": round(1 - distance, 3)}


def test_fusion_ranks_items_found_by_both_systems_first():
    vector_hits = [_hit("v-only", 0.10), _hit("both", 0.30), _hit("v-third", 0.50)]
    keyword_hits = [
        {"chunk_id": "both", "text": "t", "source": "a.pdf", "page": 1, "keyword_score": 0.9},
        {"chunk_id": "k-only", "text": "t", "source": "a.pdf", "page": 2, "keyword_score": 0.4},
    ]
    fused = hybrid.fuse(vector_hits, keyword_hits)
    assert fused[0]["chunk_id"] == "both"
    assert fused[0]["matched_by"] == "keyword+vector"


def test_fusion_keeps_items_from_a_single_system():
    fused = hybrid.fuse([_hit("v", 0.2)], [{"chunk_id": "k", "text": "t", "source": "a.pdf",
                                            "page": 3, "keyword_score": 0.5}])
    ids = {item["chunk_id"] for item in fused}
    assert ids == {"v", "k"}


def test_guard_accepts_a_close_vector_match():
    assert hybrid.should_accept(_hit("a", 0.20), "what is not patentable", []) is True


def test_guard_rejects_a_distant_match_without_lexical_evidence(monkeypatch):
    monkeypatch.setattr(config, "MAX_DISTANCE", 0.35)
    hit = _hit("a", 0.55, text="the patent office shall publish the application")
    assert hybrid.should_accept(hit, "can I patent software", ["computer programme per se"]) is False


def test_guard_accepts_a_verbatim_statutory_match_despite_a_weak_score(monkeypatch):
    monkeypatch.setattr(config, "MAX_DISTANCE", 0.35)
    hit = _hit("a", 0.55, text="(k) a computer programme per se or algorithms;")
    assert hybrid.should_accept(hit, "can I patent software", ["computer programme per se"]) is True


def test_guard_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, "MAX_DISTANCE", 0.35)
    monkeypatch.setattr(config, "LEXICAL_ACCEPT", False)
    hit = _hit("a", 0.55, text="(k) a computer programme per se or algorithms;")
    assert hybrid.should_accept(hit, "can I patent software", ["computer programme per se"]) is False


def test_guard_handles_no_hits():
    assert hybrid.should_accept(None, "anything", []) is False


def test_a_two_word_phrase_is_not_decisive_evidence(monkeypatch):
    """'trade mark' appears on half the pages, so it cannot certify a passage."""
    monkeypatch.setattr(config, "MAX_DISTANCE", 0.35)
    hit = _hit("a", 0.80, text="the trade mark shall be renewed on payment of fee")
    assert hybrid.accept_level(hit, "trademark renewal", ["trade mark"]) == 0


def test_a_three_word_phrase_is_decisive_evidence(monkeypatch):
    monkeypatch.setattr(config, "MAX_DISTANCE", 0.35)
    hit = _hit("a", 0.80, text="(k) a computer programme per se or algorithms")
    assert hybrid.accept_level(hit, "can I patent software", ["computer programme per se"]) == 2


def test_exact_phrase_hits_are_floated_to_the_top():
    fused = [
        {"chunk_id": "noise", "text": "patent application form", "rrf": 0.030, "matched_by": "vector"},
        {"chunk_id": "phrase", "text": "a computer programme per se or algorithms", "rrf": 0.010,
         "matched_by": "keyword"},
    ]
    out = hybrid.prefer_exact_phrase(fused, "computer programme per se")
    assert out[0]["chunk_id"] == "phrase"
    assert "phrase" in out[0]["matched_by"]


def test_exact_phrase_tier_ignores_short_queries():
    fused = [{"chunk_id": "a", "text": "trade mark rules", "rrf": 0.02, "matched_by": "vector"},
             {"chunk_id": "b", "text": "patents", "rrf": 0.03, "matched_by": "vector"}]
    out = hybrid.prefer_exact_phrase(fused, "trade mark")
    assert [item["chunk_id"] for item in out] == ["a", "b"]   # order untouched
