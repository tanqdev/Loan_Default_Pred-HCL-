# Loan Default Prediction — Day 1 Summary

**Domain:** Banking  
**Use Case:** Bank Muscat  
**Project:** Loan Default Prediction System  
**Date:** 2026-09-28

---

# Part 1 — Concise Summary

## 1. Project Setup

- Confirmed the project scope: ML-driven loan default prediction for a Bank Muscat banking use case.
- Confirmed the main stack:
  - Python 3.13.7
  - scikit-learn
  - XGBoost
  - SHAP
  - FastAPI
  - PostgreSQL
  - MLflow
  - Evidently AI
  - Docker
  - AWS ECS
- Decided to use a Python virtual environment (`.venv`) rather than Anaconda.
- Created the basic project structure and `requirements.txt`.

## 2. Documentation

- Prepared the SRS requirements and finalized the technology-stack section using the format:
  - `Technology : What it does`
- Created an expanded project synopsis containing:
  - Title
  - Purpose
  - Technology stack
  - System architecture
  - Frontend pages/views
  - Database schema
  - API endpoints
  - Summary
  - Conclusion
- The project documentation treats Bank Muscat as the use-case/domain context unless official Bank Muscat data is provided.

## 3. Dataset Setup

Selected the **Give Me Some Credit** dataset as the development dataset because it is a credit-risk/default prediction dataset with a binary delinquency target and AUC-oriented evaluation.

Files used:

- `cs-training.csv` — 150,000 labeled records
- `cs-test.csv` — 101,503 records with the target column present but entirely `NaN`
- `Data Dictionary.xls` — feature definitions

Important distinction:

- `cs-training.csv` is the labeled dataset used for model development.
- `cs-test.csv` is an unlabeled inference/submission dataset and cannot be used to calculate our own ROC-AUC.
- Later, the labeled training set will be split into training and validation data.
- The test file will remain untouched until final prediction.

## 4. EDA Completed

We loaded the training data into a Pandas DataFrame and investigated:

- Dataset shape
- Column names
- Target distribution
- Missing values
- Unique values
- Summary statistics
- Feature distributions
- Default vs. non-default medians and means
- Delinquency behavior
- Outliers
- Data-dictionary definitions

### Key findings

- Training data: **150,000 rows, 12 columns**.
- Target: `SeriousDlqin2yrs`.
- Target distribution:
  - `0`: 139,974
  - `1`: 10,026
  - Default/positive rate: approximately **6.68%**.
- `Unnamed: 0` is an index-like identifier and is not intended as a model feature.
- Missing values:
  - `MonthlyIncome`: 29,731
  - `NumberOfDependents`: 3,924
- `age = 0` was found once and treated as invalid.
- 13 records have ages from 100–109; these were not automatically deleted.
- `MonthlyIncome = 0` occurs 1,634 times and was left for further consideration.
- Delinquency columns contain suspicious values `96` and `98`.
- Exactly 269 rows contain `96`/`98` anomalies in all three delinquency columns.
- Those anomalous rows have a much higher default rate (~54.65%) than the overall dataset (~6.68%), so they were not deleted.
- `DebtRatio` and `RevolvingUtilizationOfUnsecuredLines` have very heavy/extreme right tails.
- Missing `MonthlyIncome` is strongly associated with much more extreme `DebtRatio` values.
- Missing income itself did not show a higher default rate in this dataset:
  - Income available: ~6.95%
  - Income missing: ~5.61%

## 5. Cleaning Work Completed

Created a separate working DataFrame:

```python
df_clean = df.copy()
```

Changes made:

- Removed `Unnamed: 0` from the modeling dataset.
- Added `DelinquencyAnomaly`:
  - `1` = at least one delinquency feature contained `96` or `98`
  - `0` = no such anomaly
- Replaced delinquency values `96` and `98` with `NaN`.
- Replaced `age = 0` with `NaN`.
- Added `IncomeMissing` as an exploratory/candidate feature:
  - `1` = `MonthlyIncome` was originally missing
  - `0` = income was present

No broad imputation or arbitrary outlier deletion was performed yet.

## 6. ML Concepts Covered

We discussed:

- Why EDA comes before modeling.
- Why mean and median can tell different stories when outliers exist.
- Class imbalance and why accuracy alone is insufficient.
- Why suspicious values should be investigated rather than automatically deleted.
- Why anomaly indicators can preserve useful information.
- Why imputation should be fitted using training data only.
- Data leakage and why preprocessing must not use validation/test information.
- Why `cs-training.csv` should be split for development while `cs-test.csv` remains untouched.
- Why train/validation/test have different roles in a general ML workflow.

## 7. Current Status

The main EDA/data-investigation phase is approximately **90% complete**.

The overall project is roughly **25–30% complete by scope**, although much of the completed work is foundational/documentation rather than final software functionality.

### Next session

1. Finalize the train/validation split using stratification.
2. Build the scikit-learn preprocessing pipeline.
3. Train the Logistic Regression baseline.
4. Evaluate it using ROC-AUC, PR-AUC, precision, recall, F1, and a confusion matrix.
5. Move on to Random Forest and XGBoost.

---

# Part 2 — Detailed Explanation

## 1. Project Context

The project is a banking-focused **Loan Default Prediction System** intended as a Bank Muscat use-case. The goal is not simply to build a classifier, but to develop a broader ML system that can eventually:

1. Predict the probability of serious loan delinquency/default.
2. Compare multiple machine-learning models.
3. Explain individual predictions using SHAP.
4. Track experiments and models using MLflow.
5. Serve predictions through a FastAPI backend.
6. Store relevant data and prediction metadata in PostgreSQL.
7. Monitor data/prediction drift using Evidently AI.
8. Package the system using Docker.
9. Deploy the containerized service using AWS ECS.

The target model-performance goal is **ROC-AUC >= 0.85**. The stated business objective is a **30% NPA reduction**, but that is a business-impact target and cannot be presented as a guaranteed result of the ML model alone.

## 2. Development Environment

The project uses Python **3.13.7** locally. A virtual environment is being used so that project-specific dependencies remain isolated from the global Python installation.

The reasoning discussed was:

```text
Global Python installation
        |
        +-- Project A environment
        |
        +-- Loan Default project environment
```

This avoids dependency conflicts and makes the environment reproducible.

Anaconda was intentionally not added because `venv + pip + VS Code` is already sufficient for this project.

The project also uses VS Code for development. Jupyter functionality can be used through the VS Code Jupyter extension rather than opening Jupyter in a separate browser.

## 3. Project Structure

The project structure established/planned is approximately:

```text
loan-default-prediction/
|
+-- .venv/
+-- docs/
|   +-- SRS.md
|   +-- synopsis.docx
+-- data/
|   +-- raw/
|   |   +-- cs-training.csv
|   |   +-- cs-test.csv
|   |   +-- Data Dictionary.xls
|   +-- processed/
+-- notebooks/
|   +-- 01_eda.ipynb
|   +-- 02_baseline_model.ipynb       # planned
|   +-- 03_model_comparison.ipynb     # planned
|   +-- 04_shap_analysis.ipynb        # planned
+-- src/                              # reusable application code
+-- models/
+-- reports/
+-- requirements.txt
+-- README.md                         # planned/ongoing
```

A key development principle was established: notebooks are for experimentation and exploration, while reusable/production logic will eventually be moved into `src/`.

## 4. Requirements and Documentation

Before coding, the mentor's instruction was to create the **SRS first**. The SRS was structured around requirements, scope, architecture, API, database, monitoring, deployment, and acceptance criteria.

The synopsis was also expanded to include multiple pages and detailed subsections rather than fitting everything into a compact document.

The technology-stack section was written in an explanatory format:

```text
Python : primary programming language
Pandas : tabular data handling
NumPy : numerical operations
scikit-learn : preprocessing, Logistic Regression, Random Forest, evaluation
XGBoost : gradient-boosting model
SHAP : model explainability
MLflow : experiment/model tracking
FastAPI : backend REST API
PostgreSQL : persistent database
Evidently AI : drift monitoring
Docker : containerization
AWS ECS : cloud deployment
Matplotlib / Seaborn : visualization
Jupyter : exploratory analysis
```

## 5. Dataset Understanding

The development dataset is the **Give Me Some Credit** dataset.

### Training data

`cs-training.csv` contains:

- 150,000 rows
- 12 columns
- A real target column: `SeriousDlqin2yrs`

### Test/inference data

`cs-test.csv` contains:

- 101,503 rows
- The same 12-column schema
- `SeriousDlqin2yrs` exists as a column name, but all 101,503 values are `NaN`

Therefore the test set is not labeled for our local evaluation.

This distinction matters because a model's ROC-AUC requires both predictions and true outcomes. Therefore:

```text
cs-training.csv
       |
       +-- train split
       +-- validation split

cs-test.csv
       |
       +-- final inference only
```

Later, after model selection and tuning, the final model can be retrained using all labeled training data and used to generate predictions for `cs-test.csv`.

## 6. EDA Setup

The first notebook is:

```text
notebooks/01_eda.ipynb
```

The notebook began with:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
```

Why these imports were used:

- **Pandas** — load and analyze tabular data.
- **NumPy** — numerical operations and array functionality.
- **Matplotlib** — plotting.
- **Seaborn** — statistical/data visualizations.

The display settings were only for notebook readability and do not affect the ML model.

## 7. Dataset Shape and Structure

The training dataset was loaded using:

```python
df = pd.read_csv("../data/raw/cs-training.csv")
```

The shape was:

```text
(150000, 12)
```

The target and features were identified as:

```text
SeriousDlqin2yrs
RevolvingUtilizationOfUnsecuredLines
age
NumberOfTime30-59DaysPastDueNotWorse
DebtRatio
MonthlyIncome
NumberOfOpenCreditLinesAndLoans
NumberOfTimes90DaysLate
NumberRealEstateLoansOrLines
NumberOfTime60-89DaysPastDueNotWorse
NumberOfDependents
```

An additional column, `Unnamed: 0`, was also present in the raw CSV but was determined to be an index-like identifier rather than a meaningful predictive feature.

## 8. Target Analysis

The target distribution was:

```text
0 -> 139,974
1 -> 10,026
```

That corresponds to approximately:

```text
0 -> 93.316%
1 ->  6.684%
```

This is a highly imbalanced binary-classification problem.

A key lesson discussed was that a naive model predicting class 0 for everyone could get high accuracy while being practically useless for identifying risky borrowers. Therefore the project will emphasize metrics such as:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion matrix
- Calibration where appropriate

## 9. Missing-Value Analysis

The initial missing-value analysis found:

```text
MonthlyIncome              29,731 missing
NumberOfDependents          3,924 missing
```

This means roughly 19.8% of the income values are missing.

The missing values were not immediately filled because imputation should eventually be learned from the training set only. This prevents validation/test information from leaking into the training process.

## 10. Feature and Distribution Analysis

`df.describe().T` and `df.nunique()` were used to understand ranges and unique-value counts.

Several unusual observations were identified:

### Age

- Minimum age = 0
- Maximum age = 109
- One row had `age = 0`
- 13 rows had age >= 100

The age value of 0 was treated as invalid and later converted to missing. The 100–109 values were kept for further consideration rather than automatically deleted.

### MonthlyIncome

- 29,731 values are missing.
- 1,634 records have income equal to 0.

The zero-income records were not automatically converted to missing because the source data does not conclusively establish that zero means missing.

### NumberOfDependents

- 3,924 values are missing.
- One record has 20 dependents.

The value 20 was not automatically deleted because it is unusual but not proven to be invalid.

## 11. Delinquency Anomalies

The following columns represent counts of past-due events:

```text
NumberOfTime30-59DaysPastDueNotWorse
NumberOfTime60-89DaysPastDueNotWorse
NumberOfTimes90DaysLate
```

Values `96` and `98` were identified as suspicious because:

- They are far outside the normal distribution of these count features.
- The values appear systematically rather than randomly.
- Exactly 5 rows contain 96 in each of the three delinquency fields.
- Exactly 264 rows contain 98 in each of the three delinquency fields.
- Therefore 269 rows have an anomalous 96/98 value pattern across all three delinquency columns.

Their default distribution was also very different from the full population:

```text
Anomalous rows:
Default     ~54.65%
No default  ~45.35%

Entire data:
Default      ~6.68%
No default  ~93.32%
```

This was an important reason not to simply delete these rows.

The final decision made for the working dataset was:

1. Add `DelinquencyAnomaly`.
2. Replace `96` and `98` with `NaN`.
3. Preserve the original records rather than deleting them.

The reasoning is that `98` should not be interpreted as "98 genuine late payments" and replacing it with zero would be even more misleading. An anomaly indicator preserves the fact that unusual source data existed.

## 12. Comparison of Default vs. Non-Default Groups

Median comparison showed that the default group tends to have:

- Much higher typical revolving credit utilization.
- Lower median age.
- Higher median debt ratio.
- Lower median monthly income.

Delinquency medians were not informative because the count variables are extremely zero-heavy; both groups had median values around zero.

A more meaningful analysis was then performed by calculating the percentage of borrowers with at least one delinquency event.

Results:

```text
30–59 days past due:
No default -> 13.57%
Default    -> 49.72%

60–89 days past due:
No default ->  3.45%
Default    -> 27.63%

90+ days late:
No default ->  3.48%
Default    -> 34.63%
```

This showed a much stronger separation than the medians.

A key lesson was learned here: **the right statistic depends on the feature's distribution**. A median of zero does not necessarily mean a feature has no predictive value when the feature is a sparse count variable.

## 13. Mean vs. Median and Outliers

Mean and median comparisons were deliberately examined together.

For heavily skewed variables such as `DebtRatio` and `RevolvingUtilizationOfUnsecuredLines`, the mean was strongly affected by extreme values.

For example, the median showed a different relationship between default groups than the mean for some ratio features.

This reinforced the principle:

> In highly skewed data, the median may be more representative of the typical observation than the mean.

At the same time, the mean for delinquency counts was useful because it captured the larger number of past-due events in the default group.

## 14. DebtRatio and Credit-Utilization Outliers

`DebtRatio` showed an extremely heavy right tail.

Approximate quantiles observed earlier included values around:

```text
50th percentile  ~0.37
90th percentile  ~1,267
95th percentile  ~2,449
99th percentile  ~4,979
99.9th percentile ~10,613
maximum          ~329,664
```

`RevolvingUtilizationOfUnsecuredLines` also had extreme values:

```text
50th percentile  ~0.15
90th percentile  ~0.98
95th percentile  ~1.00
99th percentile  ~1.09
99.9th percentile ~1,571
maximum          ~50,708
```

These features were **not arbitrarily capped or deleted** during EDA.

The current plan is to preserve the raw observations and use an appropriate preprocessing/modeling strategy. Tree-based models such as Random Forest and XGBoost can generally work with nonlinear relationships and do not require the same type of scaling as Logistic Regression.

## 15. Relationship Between Missing Income and Extreme Ratios

An analysis comparing borrowers with and without missing income showed a strong relationship between income missingness and the extreme tails of financial-ratio variables.

For example, the median `DebtRatio` was approximately:

```text
Income available -> 0.296
Income missing   -> 1159
```

This does not prove causation, but it is a strong indication that missing income is an important part of interpreting the extreme ratio values.

The missing-income groups were also compared by target rate:

```text
Income available -> ~6.95% default
Income missing   -> ~5.61% default
```

Thus, in this dataset, missing income by itself does not appear to represent a higher observed default rate.

The project therefore treats missing-income information as a **candidate signal**, rather than assuming that missing income automatically means higher risk.

An `IncomeMissing` indicator was created as an exploratory feature candidate.

## 16. Working Clean Dataset

The current working DataFrame is `df_clean`.

The important changes are:

```python
df_clean = df.copy()

df_clean = df_clean.drop(columns=["Unnamed: 0"])

# Candidate missingness feature
df_clean["IncomeMissing"] = (
    df_clean["MonthlyIncome"].isna().astype(int)
)

# Delinquency anomaly indicator
DelinquencyAnomaly = (
    df_clean[delinquency_cols]
    .isin([96, 98])
    .any(axis=1)
)

df_clean["DelinquencyAnomaly"] = DelinquencyAnomaly.astype(int)

# Convert suspicious delinquency values to missing
df_clean[delinquency_cols] = (
    df_clean[delinquency_cols]
    .replace([96, 98], np.nan)
)

# Invalid age value
df_clean.loc[df_clean["age"] == 0, "age"] = np.nan
```

No median imputation or scaling has been applied yet.

## 17. Why Imputation Has Not Been Done Yet

We discussed data leakage in detail.

The incorrect approach would be:

```python
df_clean["MonthlyIncome"].median()
```

on the full 150,000-row dataset and then using that number everywhere.

Instead, once the training/validation split is made, the preprocessing pipeline should:

1. Learn imputation/scaling parameters from training data only.
2. Apply the learned transformation to validation data.
3. Later apply the same fitted transformation to `cs-test.csv`.

This is why `Pipeline` and `ColumnTransformer` will be important in the next stage.

## 18. Train/Validation/Test Strategy Agreed

A clarification was made after inspecting `cs-test.csv`.

We are **not** treating the supplied competition test file like a labeled test set because its target is entirely missing.

The agreed workflow is:

```text
cs-training.csv (150,000 labeled)
          |
          +-- 80% Training
          |
          +-- 20% Validation

cs-test.csv (101,503 unlabeled)
          |
          +-- Final inference only
```

The split will use `stratify=y` because the target is imbalanced.

A fixed `random_state` will be used for reproducibility.

The exact split code discussed, but not yet fully executed/verified as a final step today, is:

```python
from sklearn.model_selection import train_test_split

X = df_clean.drop(columns=[TARGET])
y = df_clean[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

## 19. Key Lessons Learned Today

### EDA is not just plotting

EDA is the process of understanding what the data means, finding quality problems, checking distributions, and discovering relationships before modeling.

### High accuracy can be misleading

Because only ~6.68% of records are positive, a model that predicts the majority class can have high accuracy while failing to identify risky borrowers.

### Don't delete anomalies automatically

An unusual value can be:

- A valid extreme observation
- An invalid measurement
- A special encoding
- A missing-value placeholder

Each requires investigation.

### Mean and median answer different questions

Median describes the typical observation more robustly under outliers, while mean can capture aggregate magnitude but can be distorted by extreme values.

### Missingness itself can contain information

A missing value does not always mean the feature has no value. The fact that a value is missing can sometimes carry predictive information, so missingness indicators can be useful candidates.

### Preprocessing must avoid leakage

Statistics used to transform validation or test data must be learned from training data only.

### Documentation is part of the project

The SRS, synopsis, EDA findings, preprocessing choices, and model decisions should explain not only **what** was done but also **why**.

---

# End-of-Day Status

## Completed

- Project setup
- Python/virtual environment setup
- Requirements planning
- SRS
- Expanded synopsis
- Dataset acquisition
- Data dictionary review
- EDA
- Missing-value investigation
- Outlier investigation
- Delinquency anomaly investigation
- Initial cleaning strategy
- Initial cleaned dataset (`df_clean`)
- Understanding of labeled vs. unlabeled train/test data

## Next Session

### Day 2 — Preprocessing + Baseline Model

1. Verify the final train/validation split.
2. Define `X` and `y` cleanly.
3. Build a `ColumnTransformer`.
4. Build a scikit-learn `Pipeline`.
5. Handle missing numeric values.
6. Scale features for Logistic Regression.
7. Train the Logistic Regression baseline.
8. Evaluate ROC-AUC, PR-AUC, precision, recall, F1, and confusion matrix.
9. Record the baseline in MLflow once the tracking stage begins.

**Current overall project estimate:** approximately **25–30% complete by scope**, with the EDA/foundation stage approximately **90% complete**.
