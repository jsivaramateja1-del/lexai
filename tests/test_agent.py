"""Tests for the agentic retrieval loop.

These run with no API key, no model and no network: the retriever is replaced
with a stub that returns scripted scores, which is exactly what you want to test
the decision logic against.
"""

import src.agent as agent
from src import config


def test_key_terms_drops_stopwords_and_keeps_content():
    terms = agent.key_terms("What is not patentable in India?")
    assert "patentable" in terms
    assert "what" not in terms and "is" not in terms and "the" not in terms


def test_key_terms_is_deduplicated_and_capped():
    terms = agent.key_terms("patent patent patent novelty novelty inventive step", limit=3)
    assert len(terms) <= 3
    assert len(set(terms)) == len(terms)


def test_plan_starts_with_the_original_question():
    plan = agent.plan_queries("what is not patentable", use_llm=False)
    assert plan[0]["query"] == "what is not patentable"
    assert plan[0]["source"] == "original"


def test_plan_adds_a_key_term_variant_without_an_api():
    plan = agent.plan_queries("what is not patentable", use_llm=False)
    assert len(plan) >= 2
    assert any(item["source"] == "key terms" for item in plan)


def test_plan_respects_the_attempt_budget():
    plan = agent.plan_queries("what is not patentable", use_llm=False)
    assert len(plan) <= config.MAX_ATTEMPTS


def test_plan_deduplicates_identical_phrasings():
    plan = agent.plan_queries("patentable", use_llm=False)
    queries = [item["query"].lower() for item in plan]
    assert len(queries) == len(set(queries))


def _stub_retriever(scores):
    """Each call to search() returns the next scripted top score."""
    calls = {"n": 0}

    def search(question, top_k=5):
        score = scores[min(calls["n"], len(scores) - 1)]
        calls["n"] += 1
        return [{"chunk_id": "x", "text": "t", "source": "a.pdf", "page": 9,
                 "distance": 1 - score, "score": score}]

    return search, calls


def test_agent_stops_at_the_first_accepted_attempt(monkeypatch):
    search, calls = _stub_retriever([0.80, 0.10])
    monkeypatch.setattr(agent.retriever, "search", search)
    out = agent.collect_evidence("what is not patentable", use_llm=False)
    assert out["accepted"] is True
    assert len(out["trace"]) == 1          # stopped immediately, no wasted attempt
    assert calls["n"] == 1


def test_agent_retries_when_the_first_attempt_is_weak(monkeypatch):
    search, calls = _stub_retriever([0.10, 0.75])
    monkeypatch.setattr(agent.retriever, "search", search)
    out = agent.collect_evidence("what is not patentable", use_llm=False)
    assert out["accepted"] is True
    assert len(out["trace"]) == 2          # first rejected, second accepted
    assert out["trace"][0]["accepted"] is False


def test_agent_gives_up_and_reports_no_evidence(monkeypatch):
    search, _ = _stub_retriever([0.10, 0.05, 0.01])
    monkeypatch.setattr(agent.retriever, "search", search)
    out = agent.collect_evidence("what is the trademark fee for a small enterprise", use_llm=False)
    assert out["accepted"] is False
    assert out["hits"]                      # keeps the closest attempt for inspection
    assert out["hits"][0]["score"] == 0.10  # the best of the rejected attempts


def test_trace_is_human_readable():
    trace = [{"step": 1, "query": "what is not patentable", "source": "original",
              "top_score": 0.672, "top_page": 9, "accepted": True}]
    text = agent.describe_trace(trace)
    assert "original" in text and "0.672" in text and "accepted" in text
