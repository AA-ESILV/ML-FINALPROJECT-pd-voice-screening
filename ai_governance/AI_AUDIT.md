# AI_AUDIT — Group XX

Required: at least 4 concrete flaws found in AI outputs (at least 1 per student), with the fix.
The entries below are real issues found in the AI-generated Week 1–2 material. Before submitting, each owner must
re-check their entry themselves and rewrite it in their own words.

| # | Where | Flaw in the AI output | Why it matters | Fix | Found by |
|---|---|---|---|---|---|
| 1 | Week 1 EDA §8 (relevance ranking) | Mutual information and AUC were computed on **all 252 subjects**, including the 51 that later became the test set. | If this ranking were reused to pick features, test information would leak into model selection (Lab 4, Q5). | The Week 1 ranking is labelled "exploration only"; Week 3 feature selection runs only inside the CV pipeline on training folds. | [Student 2] |
| 2 | Week 1 specification §9 | The split was planned with `stratify=subject_class` only, ignoring the gender confounder the same EDA had found (64 % women in controls vs 57 % men in PD). | A test set with a different gender mix could make the score depend on gender, not disease. | Week 2 split stratifies on class × gender (train 51.2 % male, test 52.9 % male). | [Student 1] |
| 3 | Week 1 specification §7 | F1 was listed as a main reporting metric with PD as positive class. | Week 2 shows the useless majority baseline gets F1 = 0.86, almost equal to B2 (0.87): F1 on the majority class cannot rank models. | Models are ranked on ROC-AUC and balanced accuracy; F1 is reported for information only. | [Student 1] |
| 4 | Data source (Week 1) | The AI downloaded the CSV from a third-party GitHub mirror because UCI was unreachable, not from the official source. | An unverified copy could be modified or truncated. | We downloaded the official file from UCI and compared shape, class counts and column names; integrity `assert`s in the Week 2 notebook stop if the data differs. **(Do this check yourself before submitting.)** | [Student 2] |
| 5 | Week 1 figure 1 | The legend of the gender chart covered the bars (male percentage label hidden). | A reader could not read the 36 % male value for controls. | Legend moved above the bars. | [Student 1] |
| 6 | Week 1 spec §2 vs data | The spec claimed speech changes "often appear early", but the dataset has no disease-stage information. | Overclaiming: we cannot test early detection with this data. | Kept as background only; limitation stated in spec §5 and must not appear as a project result. | [Student 2] |
