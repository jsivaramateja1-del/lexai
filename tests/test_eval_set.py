"""Guard the evaluation set: it is ground truth, so it must be internally valid."""

import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.json"
DOCS = ROOT / "data" / "documents"


def _payload() -> dict:
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def test_the_set_exists_and_is_large_enough():
    rows = _payload()["questions"]
    assert len(rows) >= 30
    assert sum(row["answerable"] for row in rows) >= 25
    assert sum(not row["answerable"] for row in rows) >= 3


def test_ids_are_unique_and_ordered():
    ids = [row["id"] for row in _payload()["questions"]]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_every_expected_page_exists_in_its_source_document():
    counts = {}
    for row in _payload()["questions"]:
        source = row.get("source", "Patents_Act_1970.pdf")
        if source not in counts:
            with pymupdf.open(DOCS / source) as doc:
                counts[source] = doc.page_count
        for page in row["expected_pages"]:
            assert 1 <= page <= counts[source], f"{row['id']} cites page {page} of {counts[source]} in {source}"


def test_every_question_names_a_source_that_exists():
    for row in _payload()["questions"]:
        assert (DOCS / row["source"]).exists(), f"{row['id']} points at a missing document"


def test_the_set_covers_more_than_one_statute():
    sources = {row["source"] for row in _payload()["questions"]}
    assert len(sources) >= 2, f"expected two statutes, found {sources}"


def test_answerable_questions_have_expected_pages():
    for row in _payload()["questions"]:
        if row["answerable"]:
            assert row["expected_pages"], f"{row['id']} has no expected page"


def test_unanswerable_questions_claim_no_pages():
    for row in _payload()["questions"]:
        if not row["answerable"]:
            assert row["expected_pages"] == [], f"{row['id']} should not expect a page"


def test_questions_are_real_sentences():
    for row in _payload()["questions"]:
        assert row["question"].strip().endswith("?"), row["id"]
        assert len(row["question"]) > 15
