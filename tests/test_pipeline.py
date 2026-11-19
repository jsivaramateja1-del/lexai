"""Tests for the pipeline's decision paths.

The four outcomes a question can have are tested here without a model, without a
network and without touching the query log:

    no_answer   - nothing close enough, refuse
    extractive  - quote mode (no key, or LLM disabled)
    llm         - the model answered
    llm_error   - the model was called and failed
"""

import src.pipeline as pipeline


def _stub_evidence(accepted: bool, score: float = 0.8):
    def collect(question, top_k=None, use_llm=True):
        hits = [
            {"chunk_id": "a", "text": "Section 3 text", "source": "a.pdf",
             "page": 10, "distance": 1 - score, "score": score}
        ]
        return {
            "hits": hits,
            "accepted": accepted,
            "trace": [{"step": 1, "query": question, "source": "original",
                       "top_score": score, "top_page": 10, "accepted": accepted}],
        }

    return collect


def _no_log(monkeypatch):
    monkeypatch.setattr(pipeline, "log_query", lambda row: None)


def test_refuses_when_no_evidence_is_accepted(monkeypatch):
    _no_log(monkeypatch)
    monkeypatch.setattr(pipeline.agent, "collect_evidence", _stub_evidence(False, 0.4))
    result = pipeline.answer_question("what is the trademark fee for a small enterprise")
    assert result["mode"] == "no_answer"
    assert "do not cover" in result["answer"]


def test_quote_mode_when_no_model_is_configured(monkeypatch):
    _no_log(monkeypatch)
    monkeypatch.setattr(pipeline.agent, "collect_evidence", _stub_evidence(True))
    monkeypatch.setattr(pipeline.generator, "llm_available", lambda: False)
    result = pipeline.answer_question("what is not patentable")
    assert result["mode"] == "extractive"
    assert result["sources"][0]["page"] == 10


def test_llm_mode_when_the_model_answers(monkeypatch):
    _no_log(monkeypatch)
    monkeypatch.setattr(pipeline.agent, "collect_evidence", _stub_evidence(True))
    monkeypatch.setattr(pipeline.generator, "llm_available", lambda: True)
    monkeypatch.setattr(
        pipeline.generator, "call_llm",
        lambda messages, retries=2: {"text": "Frivolous inventions are not patentable.",
                                     "latency_s": 0.4, "error": ""},
    )
    result = pipeline.answer_question("what is not patentable")
    assert result["mode"] == "llm"
    assert result["answer"].startswith("Frivolous")
    assert result["llm_ms"] == 400


def test_model_failure_is_reported_distinctly(monkeypatch):
    _no_log(monkeypatch)
    monkeypatch.setattr(pipeline.agent, "collect_evidence", _stub_evidence(True))
    monkeypatch.setattr(pipeline.generator, "llm_available", lambda: True)
    monkeypatch.setattr(
        pipeline.generator, "call_llm",
        lambda messages, retries=2: {"text": "", "latency_s": 1.0, "error": "HTTP 403"},
    )
    result = pipeline.answer_question("what is not patentable")
    assert result["mode"] == "llm_error"
    assert "HTTP 403" in result["error"]
    assert "source passages" in result["answer"]


def test_every_result_carries_the_agent_trace(monkeypatch):
    _no_log(monkeypatch)
    monkeypatch.setattr(pipeline.agent, "collect_evidence", _stub_evidence(True))
    monkeypatch.setattr(pipeline.generator, "llm_available", lambda: False)
    result = pipeline.answer_question("what is not patentable")
    assert result["trace"] and result["trace"][0]["step"] == 1
    assert "original" in result["trace_text"]


def test_empty_question_is_rejected_before_any_work(monkeypatch):
    _no_log(monkeypatch)
    result = pipeline.answer_question("   ")
    assert result["mode"] == "error"


def test_pipeline_trusts_the_agents_lexical_acceptance(monkeypatch):
    """A weak cosine score plus a statutory phrase must still be answered."""
    _no_log(monkeypatch)
    weak = {"chunk_id": "a", "text": "(k) a computer programme per se", "source": "a.pdf",
            "page": 10, "distance": 0.66, "score": 0.34}
    monkeypatch.setattr(
        pipeline.agent, "collect_evidence",
        lambda question, top_k=None, use_llm=True: {
            "hits": [weak], "accepted": True, "strong": True,
            "trace": [{"step": 1, "query": question, "source": "original",
                       "top_score": 0.34, "top_page": 10, "level": 2, "accepted": True}],
        },
    )
    monkeypatch.setattr(pipeline.generator, "llm_available", lambda: False)
    result = pipeline.answer_question("can I patent software")
    assert result["mode"] == "extractive"      # answered, not refused
    assert result["sources"][0]["page"] == 10
