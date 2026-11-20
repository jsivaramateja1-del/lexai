# Power BI dashboard — build steps

The data is prepared for you; this is the recipe to turn it into the graded dashboard. Power BI
Desktop is Windows-only, which is what you have, so nothing here needs a licence — the free
desktop app is enough to build and screenshot.

## 1. Produce the data

```bash
python -m scripts.export_dashboard     # analytics/dashboard_data.csv
python -m scripts.run_eval             # eval/results.csv (if you have not run it recently)
```

`dashboard_data.csv` has one row per question with derived columns already computed, so the
report never repeats the same DAX in five places:

| Column | Meaning |
| --- | --- |
| `date`, `hour_of_day` | when the question was asked |
| `question` | the text the user typed |
| `answer_mode` | `llm`, `extractive`, `llm_error` or `no_answer` |
| `answered`, `refused` | 1/0 flags, ready for summing |
| `source_doc`, `top_page` | the document most cited, and its first page |
| `top_distance` | the best cosine distance — how close the evidence was |
| `attempts` | how many search phrasings the agent tried |
| `retrieval_ms`, `llm_ms`, `total_ms` | timings |
| `latency_band` | `<100 ms`, `100-300 ms`, `300-1000 ms`, `>1000 ms` |
| `feedback` | `up`, `down` or blank |

## 2. Load both tables

1. Open Power BI Desktop → **Get data** → **Text/CSV**.
2. Load `analytics/dashboard_data.csv`, then `eval/results.csv` as a second query.
3. In **Transform Data**, set types: `date` → Date, `answered`/`refused` → Whole number,
   `total_ms`/`retrieval_ms`/`llm_ms` → Decimal, `top_distance` → Decimal.
4. **Close & Apply.**

## 3. Measures

Create these under **Modeling → New measure**. They are what makes the report defensible:
each one is a sentence you can say out loud.

```dax
Total Questions = COUNTROWS(dashboard_data)

Answered = SUM(dashboard_data[answered])

Refused = SUM(dashboard_data[refused])

Refusal Rate = DIVIDE([Refused], [Total Questions])

Median Latency (ms) = MEDIAN(dashboard_data[total_ms])

Mean Attempts = AVERAGE(dashboard_data[attempts])

Citation Coverage =
DIVIDE(
    CALCULATE(COUNTROWS(dashboard_data), dashboard_data[num_sources] > 0, dashboard_data[answered] = 1),
    [Answered]
)

Evaluation Hit Rate =
DIVIDE(SUM(results[retrieval_hit]), COUNTROWS(FILTER(results, results[answerable] = TRUE)))

Evaluation Top-1 = DIVIDE(SUM(results[top1_correct]), COUNTROWS(FILTER(results, results[answerable] = TRUE)))
```

## 4. The five visuals

Put these on one page. Four are required; the fifth is the one that impresses.

| # | Visual | Fields | The point it makes |
| --- | --- | --- | --- |
| 1 | Card ×3 | `Total Questions`, `Refusal Rate`, `Median Latency (ms)` | the system's operating state at a glance |
| 2 | Line chart | Axis `date`, Values `Total Questions` | usage over time |
| 3 | Column chart | Axis `answer_mode`, Values `Total Questions` | how often it answers vs refuses, and how often the model failed |
| 4 | Bar chart | Axis `source_doc` + `top_page`, Values `Total Questions` (Top N = 10) | which sections of which Act the questions actually come from |
| 5 | Column chart | Axis `latency_band`, Values `Total Questions`, plus a line for `Mean Attempts` | where the time goes, and how much the agentic retry costs |

Add slicers for `date`, `source_doc` and `answer_mode`. Add a title that states the finding, not
the topic: *"Questions are answered 78% of the time; refusals cluster on trademark fees"* beats
*"Dashboard"*.

## 5. A second page for the evaluation

Use `eval/results.csv`: a table of the 39 questions with `retrieval_hit`, `top1_correct`,
`refused` and `best_distance`, plus the two evaluation measures above. Colour `retrieval_hit = 0`
rows so the misses are visible. This page is where you answer "how do you know it works?".

## 6. Finish

- Save as `LexAI-dashboard.pbix` in the repo root (or `docs/`), commit it if it is under a few MB.
- **File → Export → PDF** and put the pages in your report.
- Screenshot each page for the README and the slides.

## Honest note for the report

The dashboard describes usage, not correctness. Correctness lives on page two, in the evaluation
table, and the two should never be presented as the same thing: a hundred questions answered is
not a hundred questions answered *well*.
