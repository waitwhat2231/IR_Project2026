# Evaluation Service (Requirement 8) — Simple Guide

This document explains, in plain language, the **Ranking & Evaluation Service**
that was added to the project. It is written for someone who is **not** a Python
expert, so every part is explained with small examples.

---

## 1. What is this and why do we need it?

After we built several search models (TF-IDF, BM25, SBERT, Word2Vec, and the
Hybrid models), we need to answer one question:

> **"Which search model gives the best results?"**

To answer this fairly, we use **standard IR (Information Retrieval) metrics**.
We run every model on the same test questions and measure how good its answers are.

The project requires these 4 metrics:

| Metric                           | In one sentence                                                          |
| -------------------------------- | ------------------------------------------------------------------------ |
| **MAP** (Mean Average Precision) | Are the correct documents ranked near the top, on average?               |
| **Recall**                       | Out of all correct documents, how many did we find?                      |
| **Precision@10**                 | Of the first 10 results, how many are actually correct?                  |
| **nDCG**                         | Like precision, but also rewards putting the _most_ relevant docs first. |

A higher number is always better (all values are between 0 and 1).

---

## 2. The ingredients (the data we compare against)

To grade a model we need an **answer key**. The dataset gives us two files:

### a) Queries — the test questions

File: `data/raw/webis-touche2020/queries.json`

```json
{
  "1": "Should teachers get tenure?",
  "2": "Is vaping with e-cigarettes safe?"
}
```

### b) Qrels — the answer key (which documents are correct)

File: `data/raw/webis-touche2020/qrels.json`

`qrels` = "query relevance judgments". For each question, experts already marked
which documents are relevant and how relevant they are:

```json
{
  "1": {
    "doc_A": 2, // very relevant
    "doc_B": 1, // relevant
    "doc_C": 0 // not relevant
  }
}
```

- `0` = not relevant
- `1` = relevant
- `2` = highly relevant

> In our code, a document counts as "correct" when its grade is **1 or higher**.

---

## 3. The 3 files we created

Everything lives in the folder:
`Services/ranking_evaluation_service/`

| File           | Job                                          | Think of it as...  |
| -------------- | -------------------------------------------- | ------------------ |
| `scorer.py`    | Does the **math** of the metrics             | The calculator     |
| `evaluator.py` | Runs each model and **collects** the results | The manager        |
| `main.py`      | The **start button** you actually run        | The remote control |

This separation is on purpose: the math, the orchestration, and the entry point
are kept apart so each piece is easy to read, test, and reuse.

---

## 4. `scorer.py` — the calculator (the math)

This file only does math. It takes two things:

1. **The model's answer** = a ranked list of documents it returned, best first:
   ```python
   ranked = [("doc_C", 9.1), ("doc_A", 8.7), ("doc_X", 4.0)]
   ```
2. **The answer key** for that question:
   ```python
   relevant = {"doc_A": 2, "doc_B": 1}   # doc_A and doc_B are the correct ones
   ```

Then it calculates the metrics. A tiny example for **Precision@2**
(of the first 2 results, how many are correct?):

```text
ranked top 2 = doc_C, doc_A
correct ones = doc_A, doc_B
doc_C -> wrong
doc_A -> correct
=> 1 correct out of 2  =>  Precision@2 = 0.5
```

The functions inside `scorer.py`:

- `precision_at_k(...)` → Precision@k
- `recall_at_k(...)` → Recall@k
- `average_precision(...)` → the per-question score used to build MAP
- `ndcg_at_k(...)` → nDCG@k
- `score_run(...)` → runs all of the above for **every** question and then
  **averages** them to give one final number per metric.

> **Why "average"?** Each question gets its own score first. The final number you
> see in the table is the average across all 49 questions. (That is literally what
> the "M" in **M**AP means: _Mean_ Average Precision.)

---

## 5. `evaluator.py` — the manager

This file does 4 things:

1. **Loads the data once** (the queries and the answer key).
2. **Asks each model for its answers** to all 49 questions. Important detail:
   each model wants its input in a different shape, and the manager handles that:

   | Model    | What we feed it                                          |
   | -------- | -------------------------------------------------------- |
   | TF-IDF   | the cleaned/stemmed query text                           |
   | BM25     | the cleaned query as a list of words                     |
   | Word2Vec | the cleaned query as a list of words                     |
   | SBERT    | the **original** query text (it cleans text its own way) |
   | Hybrid   | both at the same time                                    |

3. **Calls the calculator** (`scorer.py`) to grade those answers.
4. **Saves everything** neatly to disk (explained in section 7).

It also measures **how long** each model takes (speed matters too).

---

## 6. `main.py` — the start button

This is the file you run. In order, it:

1. Loads all the models into memory (once, to save memory).
2. Evaluates each model one by one.
3. Prints a comparison table on the screen.
4. Saves the detailed results into files.

### How to run it

Open a terminal in the project root and run:

```powershell
conda activate ir_project
python Services/ranking_evaluation_service/main.py
```

> **Tip (Windows):** if you see a weird `charmap` / encoding error, run it like
> this instead (it just tells Python to use UTF-8 for the screen output):
>
> ```powershell
> $env:PYTHONIOENCODING="utf-8"
> & "C:\Users\Loukas\anaconda3\envs\ir_project\python.exe" Services/ranking_evaluation_service/main.py
> ```

### Optional settings

You don't need these, but they exist:

```powershell
# evaluate only some models
python Services/ranking_evaluation_service/main.py --models tfidf,bm25

# change how deep we search (default is 1000)
python Services/ranking_evaluation_service/main.py --top_k 1000

# label the run (more on this in section 8)
python Services/ranking_evaluation_service/main.py --phase baseline
```

---

## 7. Where the results are saved

After running, everything is saved here:

```
data/evaluation/webis-touche2020/baseline/
├── metrics_summary.json     <- the full summary (numbers + info)
├── comparison.csv           <- a table you can open in Excel
├── report.txt               <- a human-readable report
└── per_query/
    ├── tfidf.json           <- score of each question for TF-IDF
    ├── bm25.json
    ├── sbert.json
    ├── word2vec.json
    ├── hybrid_parallel.json
    └── hybrid_serial.json
```

- **`comparison.csv`** is the easiest one — open it in Excel for the report.
- **`report.txt`** also shows, question by question, which model did best.
- **`per_query/*.json`** is for deep analysis of a single model.

---

## 8. The "phase" idea (before vs. after extra features)

The project asks us to evaluate **twice**:

- **before** adding the extra/bonus features, and
- **after** adding them,

so we can prove the extra features actually helped.

To support this, every run has a **phase** label:

```powershell
# current run = the "before" baseline
python Services/ranking_evaluation_service/main.py --phase baseline

# later, after you add features = the "after" run
python Services/ranking_evaluation_service/main.py --phase enhanced
```

Each phase saves into its own folder
(`.../baseline/` and `.../enhanced/`), so you can compare them side by side.

---

## 9. The results we got (baseline)

This is the actual output from running it on **49 test questions**:

| model           | MAP        | Recall@1000 | P@10       | nDCG@10    |
| --------------- | ---------- | ----------- | ---------- | ---------- |
| tfidf           | 0.0520     | 0.7330      | 0.0673     | 0.0566     |
| **bm25**        | **0.2194** | 0.8726      | **0.2898** | **0.3172** |
| sbert           | 0.1312     | 0.7706      | 0.1673     | 0.1709     |
| word2vec        | 0.1017     | 0.7441      | 0.1347     | 0.1561     |
| hybrid_parallel | 0.2137     | **0.8784**  | 0.2653     | 0.2903     |
| hybrid_serial   | 0.1431     | 0.8708      | 0.1755     | 0.1781     |

### How to read this

- **BM25 is the best overall** model on this dataset (highest MAP, P@10, nDCG@10).
- **hybrid_parallel** finds the most correct documents in total (highest Recall)
  and is almost as good as BM25.
- **TF-IDF** is the weakest here.
- This kind of comparison is exactly what the project report needs.

> Remember: numbers are between 0 and 1, and **higher is better**.

---

## 10. Quick summary

- We added a service that **grades and compares** all search models fairly.
- It uses the 4 required metrics: **MAP, Recall, Precision@10, nDCG**.
- Run it with **one command**; results are saved as **CSV, JSON, and a text report**.
- It supports **before/after** comparison through the `--phase` option.
