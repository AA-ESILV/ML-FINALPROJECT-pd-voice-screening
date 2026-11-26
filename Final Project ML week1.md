*Parkinson’s disease classification based on voice data.*

***Team Members :*** ***Armelle ADAM, Lola COCHE***

1. **Context and Motivation :** 
Parkinson’s disease is a progressive neurodegenerative disorder and changes in speech like dysphonia, reduced loudness or unstable pitch often appear early. Today the diagnosis is mostly clinical. A neurologist observes the motor symptoms, which is subjective and usually happens late in the disease.

A voice recording is cheap, non-invasive and can be done with a simple phone microphone. With signal processing, it can be turned into numerical biomarkers such as jitter, shimmer, harmonicity, MFCCs, and wavelet and TQWT coefficients. That is basically what medical engineering is about, going from a sensor to a signal, then to features and finally to a model that helps with decisions.

What motivates us is building a screening aid rather than a diagnostic tool, so something that tells people when they should go see a neurologist. Missing a patient (false negative) is worse than sending someone to an extra appointment (false positive).

1. **Problem Statement :** 
Our dataset has almost as many features (753) as recordings (756) and each person was recorded three times. Because of this, it is easy to end up with a model that looks very good in validation but doesn’t work on a new patient.

Research question: using only the acoustic features of a sustained vowel, can a classical classifier detect Parkinson’s disease in someone it has never heard before, and how many of the 753 features does it really need?

From this, we ask two sub-questions:

- Which type of feature selection (filter, wrapper or embedded) gives the best balance between recall, stability and number of features?

- How much does the score go down when we evaluate per person instead of per recording?

1. **Project Objectives :**


|  | Objective | Target |
| --- | --- | --- |
| O1 | Do better than a trivial model | Balanced accuracy clearly higher than 0.50 which is what predicting the majority class gives |
| O2 | Good enough for screening | ROC-AUC ≥ 0.85 and PD recall ≥ 0.90 with the threshold we choose |
| O3 | Keep the model small | 50 features or fewer, with a CV score less than one standard deviation below the model using all features |
| O4 | Evaluate it properly | Compare recording-level and subject-level CV and don’t touch the test set before Week 4 |
| O5 | Understand what the model uses | Identify which feature groups (MFCC, TQWT, jitter…) the final model relies on, and how often they get selected over bootstrap resamples |
We know the O2 targets are high, but previous studies on this dataset reached around 0.85 accuracy using feature selection, so they seem achievable. We might adjust them once we have our baseline in Week 2.

1. **Dataset Description :** 
We use the [UCI Parkinson’s Disease Classification dataset](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification) (Sakar et al., 2018, CC BY 4.0, DOI 10.24432/C5MS4X). The data was collected at Istanbul University. Each participant said the vowel /a/ three times, holding it, and features were extracted from every recording.

- Size: 756 rows (one per recording) and 755 columns, which are id (the subject), gender, 752 acoustic features and the target class.

- Subjects: 252 people with 3 recordings each. 188 have PD (107 men and 81 women) and 64 are healthy controls (23 men and 41 women).

- Target: class is 1 for PD and 0 for healthy, and it stays the same for all recordings of a given subject.

- Types and missing values: everything is numerical (749 floats and 4 integers), and gender is binary with 1 for male. There are no missing values

The 752 features come from 8 groups: baseline dysphonia measures (21), intensity (3), formants (4), bandwidth (4), vocal fold (22), MFCC (84), wavelet (182) and TQWT (432).

It’s a real biomedical signal problem, the data is clean with no missing values, and d ≈ n, which is exactly the case where feature selection is useful. It’s also small enough to run SVM and RFE on a laptop.

Limitations:

- The data comes from one centre and one language (Turkish speakers) so the results may not hold for other populations or microphones.

- The sample is small (252 people) so CV estimates will have a high variance.

- Gender is unbalanced between classes, with 64% women among controls and 57% men among patients. The model could partly learn gender rather than the disease.

- There is no information on disease stage or medication so we can’t study early detection specifically.

1. **Machine Learning Task :**
This is a supervised binary classification problem and the target seems moderately imbalanced.

- Input X: the 752 acoustic features of a recording, plus gender. We won’t use id as a feature, it will only serves to group the recordings when we will split the data.

- Model output: a probability P(PD | x) for each recording, which we will convert into a label using a decision threshold.

- Subject-level decision: we will average the probabilities of a person’s three recordings to get one prediction per person, because clinically the question is about the person and not the recording.

**Metrics:** our main metric will probably be ROC-AUC, since it doesn’t depend on a threshold and  should handle the 2.9 : 1 ratio well. We will also report PD recall (sensitivity, which is the priority for screening), specificity, F1, balanced accuracy and the confusion matrix. 

1. **Baseline Strategy :**
We will use two baselines, both set up in Week 2 before any other model.

1. Majority-class classifier (DummyClassifier): always predicts PD. It gives the minimum score to beat and shows why accuracy alone is misleading.

2. Logistic regression on all features: Pipeline(StandardScaler → LogisticRegression(class_weight="balanced", max_iter=1000)), with no feature selection and the default C.

We chose logistic regression because it’s the simplest probabilistic linear model, fast and interpretable. Scaling is needed since feature scales differ by up to 10^29 and class_weight="balanced" will handle the 2.9 : 1 ratio without synthetic samples. With 753 features for about 200 training subjects, we expect it to overfit which is what Week 3 should fix.

1. **Evaluation Strategy :**
We always split by subject and not by recording, so that a person’s three recordings stay in the same set. The EDA shows why this matters: for 68% of recordings, the closest other recording comes from the same person.

1. Test set: 20% of subjects (about 51 people), picked once with train_test_split(..., stratify=subject_class, random_state=42) on the subject ids and left untouched until Week 4.

2. Training and validation: the other 80% (about 201 people), using 5-fold StratifiedGroupKFold with groups=id. This keeps the class proportions and puts each person in a single fold, so each fold is used as validation in turn.

3. No leakage: No leakage: scaling, feature selection and resampling are inside the Pipeline, so they’re only fitted on the training folds. In Week 2 we also run a normal StratifiedKFold once to show how much leakage inflates the score.

4. Stability: we repeat the CV with 3 seeds and report the mean ± standard deviation, since n is small.

1. **Initial Work Plan :** 


| Week | Main tasks | Armelle ADAM | Lola COCHE | Deliverable |
| --- | --- | --- | --- | --- |
| Definition and Data | Topic, dataset, EDA, specification | EDA notebook, figures | Specification, literature review | This document |
| Preparation and baseline | Subject-level split, Pipeline, dummy and logistic baselines, leakage check | Split and pipeline | Baselines and metrics | Split and baseline results |
| Models and experiments | kNN, SVM-RBF, tree, filter/wrapper/embedded selection, class_weight vs SMOTE, GridSearchCV | SVM, kNN, filters | Tree, RFE, L1, stability | Comparison with baseline |
| Evaluation and presentation | Test-set evaluation, confusion matrix, threshold, error analysis, report | Results, figures | Report, slides | Report, notebook, oral |
**Main risks:** the small sample (252 subjects) makes results unstable, so we will repeat the grouped CV. The model could learn gender, so we will check recall per gender and test a model without it. RFE on 750 features may be slow, so we will filter down to about 100 first.
