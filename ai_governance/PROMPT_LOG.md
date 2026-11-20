# PROMPT_LOG — Group XX

Required: 4–6 high-stakes LLM interactions (at least 1 per student), showing how the prompt constrained the model
(schemas, assumptions, leakage guardrails). Entries 1–2 are the real prompts we used so far; entries 3–4 are the
constrained prompts we used for the Week 2 code. Adapt wording to what you actually typed, and add Week 3–4 entries.

---
## Entry 1 — Topic and dataset choice (Week 1) — owner: [Student 1]
**Prompt (summary):** "Act as a 4th-year medical-engineering student following these 4 lessons [attached]. Give 5 project ideas with
dataset links and the methods each would use. Think about later steps and doability. Do not use concepts that are not in the lessons."
**Constraints given:** course scope only; real dataset links; feasibility over 4 weeks.
**What we kept / rejected:** kept idea 2 (Parkinson voice); we verified every link opened and that the dataset counts
(756 rows, 252 subjects, 188/64) matched the UCI page.

## Entry 2 — Week 1 specification and EDA — owner: [Student 2]
**Prompt (summary):** "Do idea 2: write the Week 1 specification with the 10 required sections and an EDA notebook."
**Constraints given:** the 10 sections of the lab; English; 2–3 pages.
**Verification:** re-ran the notebook ourselves; checked each number in the spec against the notebook output.

## Entry 3 — Split and CV protocol (Week 2) — owner: [Student 2]
**Prompt (summary):** "Write the train/test split and CV code. Constraints: (1) split on subject `id`, never on rows;
(2) test = 20 % of subjects, stratified on class × gender, `random_state=42`, saved to `split.csv`;
(3) CV = `StratifiedGroupKFold(5)` repeated with seeds 0,1,2 on training subjects only; (4) no transformer fitted
outside the `Pipeline`; (5) the test set must not be read anywhere."
**Verification:** `assert set(train_ids).isdisjoint(test_ids)`; checked no subject appears in two folds;
searched the notebook for `test_ids` (used only to build `split.csv`).

## Entry 4 — Baselines and metrics (Week 2) — owner: [Student 1]
**Prompt (summary):** "Implement B0 majority, B1 gender-only, B2 StandardScaler + LogisticRegression(class_weight='balanced').
Report ROC-AUC, recall PD, specificity, balanced accuracy, F1, accuracy at recording AND subject level
(subject = mean of 3 probabilities). Output a mean ± std table over 15 folds. Also report train vs validation AUC."
**Verification:** recomputed B0 by hand (accuracy = 150/201 ≈ 0.746, AUC = 0.5); checked confusion-matrix totals = 201 subjects.

## Entry 5 — (Week 3, to fill) — owner:
## Entry 6 — (Week 4, to fill) — owner:
