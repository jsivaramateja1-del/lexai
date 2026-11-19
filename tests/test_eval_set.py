"""Guard the evaluation set: it is ground truth, so it must be internally valid."""

import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.json"
PDF = ROOT / "data" / "documents" / "Patents_Act_1970.pdf"


def _payload() -> dict:
    return json.loads(QUESTIONS.read_text(encoding="utf-8"))


def test_the_set_exists_and_is_large_enough():
    rows = _payload()["questions"]
    assert len(rows) >= 20
    assert sum(row["answerable"] for row in rows) >= 15
    assert sum(not row["answerable"] for row in rows) >= 3


def test_ids_are_unique_and_ordered():
    ids = [row["id"] for row in _payload()["questions"]]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)


def test_every_expected_page_exists_in_the_pdf():
    with pymupdf.open(PDF) as doc:
        page_count = doc.page_count
    for row in _payload()["questions"]:
        for page in row["expected_pages"]:
            assert 1 <= page <= page_count, f"{row['id']} cites page {page} of {page_count}"


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
