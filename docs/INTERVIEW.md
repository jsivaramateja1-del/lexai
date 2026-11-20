# Interview pack

Everything here is drawn from the project's own measurements. Where a number is missing, it says so
rather than inventing one.

## Numbers cheat-sheet

| Fact | Value |
| --- | --- |
| Corpus | 2 statutes, 120 pages |
| Chunks | 629 (900 characters, 150 overlap, page-scoped, word-aligned) |
| Embedding model | paraphrase-multilingual-MiniLM-L12-v2, 384 dimensions, local |
| Keyword index | TF-IDF, 1–3 word n-grams, fused with vectors by reciprocal rank |
| Refusal threshold | cosine distance 0.40 (strong at 0.28) |
| Retrieval hit@5 | 75% on 29 questions (re-measure after the second statute) |
| Top-1 correct | 41.7% — several questions have three valid pages |
| Citations | 100% of answered questions |
| Median retrieval latency | 81 ms, no network |
| Agent attempts | mean 1.6, capped at 4 |
| Tests | 64 unit tests, 41 system checks, 39-question evaluation |
| Cost | about 2 paise per answered question |

## The eight questions you will actually be asked

**1. Why RAG instead of fine-tuning the model?**
Fine-tuning changes how a model writes; it does not give you a citation. I need the answer to point
at page 30 of a specific Act, and I need it to change the moment the document changes. Retrieval gives
both. Fine-tuning would also bake in a version of the law that silently ages.

**2. Isn't this just a wrapper around an API?**
The API is one file and about forty lines. The other 90% is ingestion, chunking, a hybrid index, a
three-level relevance guard, an agentic retry loop, evaluation and tests. Swapping the model for
another provider changes one function.

**3. How did you choose the chunk size?**
By experiment, not preference. Three settings — 450/80, 900/150, 1400/200 — against four fixed
questions. 900/150 was the only one that returned the correct page for both answerable questions.
The 450 setting scored *highest* while returning the wrong page, which is why I stopped treating
similarity as a correctness measure.

**4. How did you choose the refusal threshold?**
By sweeping it across the evaluation set. Lowering it answers more questions and invents more
answers; the sweep shows both sides. At 0.40 the system answers 19 of 24 with one false answer,
against 12 of 24 at 0.30. Zero false answers is unreachable on this set, and I say so.

**5. What makes the retrieval "agentic"?**
It plans several search phrasings — the question, a statutory-wording variant from a glossary, a
key-term variant, and a model rewrite — then acts, observes the grade of the evidence, and re-acts
while the grade is not strong, capped at four attempts. Every attempt is recorded and shown in the
interface. It stops on strong evidence, not on the first result.

**6. Why a keyword index as well as vectors?**
A statute says "computer programme per se" where a user says "software". Dense vectors are good at
meaning and bad at exactness; TF-IDF is the reverse. The two are merged by reciprocal rank, which
compares ranks rather than scores, so a cosine distance and a TF-IDF weight never need a shared scale.

**7. What breaks it today?**
Two off-topic questions still get answered, because refusal is a threshold rather than a proof. It
knows two statutes of four. PDF extraction leaves a few column-join artefacts. And faithfulness is
measured only by citation presence, not by checking every claim against its passage.

**8. If you had four more weeks?**
The Copyright Act and the examination guidelines; a faithfulness check that compares each claim with
the passage it cites; hybrid search with a proper BM25 implementation rather than TF-IDF; and the
Power BI dashboard on the query log, which already exists as data.

## The three uncomfortable questions

**"Your top-1 accuracy is 41.7%. Is that bad?"**
It is the weakest number, and partly an artefact of the test: several questions have three acceptable
pages, so top-1 measures ranking luck as much as skill. The number that matters more is hit@5 at 75%,
because if the right page is in the retrieved set, the model can cite it — and citation coverage is
100%.

**"You said it refuses off-topic questions, but two got through."**
Correct, and it is in the report. The threshold bounds the damage; it does not eliminate it. One of
those questions is about trademark fee amounts, which live in the Trade Marks Rules rather than the
Act, so a better knowledge base fixes it rather than a better threshold.

**"How do you know the answers are right?"**
I don't, in the general sense — I know each answer is checkable. Every claim carries a document and a
page, and the interface shows the passage. On the evaluation set I can tell you how often the right
page was retrieved and how often the system answered at all. What I have not measured is faithfulness
claim by claim, and that is the next thing I would build.

## Resume bullets

Use these as written once you have re-run the evaluation; replace the marked figures with the new
ones.

- Built a retrieval-augmented assistant over Indian IP statutes (2 documents, 120 pages, 629 chunks)
  that answers only from retrieved passages and cites document and page on 100% of answers.
- Designed hybrid retrieval — sentence-transformer vectors fused with a TF-IDF keyword index by
  reciprocal rank — after measuring that dense search alone missed exact statutory phrasing such as
  "computer programme per se".
- Implemented a three-level relevance guard (strong / provisional / reject) and set its threshold by
  sweeping an evaluation set, trading answered questions against false answers instead of guessing.
- Built a bounded agentic retrieval loop that proposes up to four search phrasings, stops only on
  strong evidence, and records an auditable trace shown in the UI.
- Wrote 64 unit tests and a 41-check system test drive; six defects found this way are documented
  with their symptoms and fixes, including a chunker bug that produced 1,688 fragments from 69 pages.
- Reduced a hallucination failure mode to a measured rate: 3 of 5 off-topic questions refused, with
  the two misses named in the report rather than hidden.

## One sentence, for a cover letter

> I built a retrieval assistant that answers Indian IP questions only from official documents, cites
> the page for every claim, and refuses when the documents do not cover the question — and I measured
> how often it does each of those things.
