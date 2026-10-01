# Day 2 --- Loan Default Prediction Project

## Overview

Day 2 focused on moving from data preparation into **model development,
model evaluation, explainability, and experiment tracking**.

The main goal was to train multiple classification models, compare their
performance, tune XGBoost, analyze classification thresholds, explain
predictions using SHAP, and set up MLflow for experiment tracking.

------------------------------------------------------------------------

# 1. Train / Validation Split

The cleaned dataset was divided into training and validation sets.

``` python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```

### Why?

The validation set is kept separate from training so that we can
evaluate how well the model performs on data it has not seen during
training.

`stratify=y` was used to preserve the original class distribution
because the dataset is imbalanced.

------------------------------------------------------------------------

# 2. Logistic Regression

A Logistic Regression model was trained as a baseline.

A preprocessing pipeline was used:

``` text
Missing values
     ↓
Median Imputation
     ↓
Robust Scaling
     ↓
Logistic Regression
```

Two versions were tested:

-   Normal Logistic Regression
-   Class-balanced Logistic Regression

### Results

  -----------------------------------------------------------------------------
  Model             ROC-AUC       PR-AUC    Precision       Recall           F1
  ------------ ------------ ------------ ------------ ------------ ------------
  Logistic           0.8205       0.3599       0.5928       0.1561       0.2471
  Regression                                                       

  Logistic           0.8255       0.3629       0.2708       0.6304       0.3788
  Regression                                                       
  (Balanced)                                                       
  -----------------------------------------------------------------------------

### Observation

Class balancing increased recall significantly, but precision decreased.

------------------------------------------------------------------------

# 3. Random Forest

Random Forest was then tested.

Two versions were evaluated:

-   Normal Random Forest
-   Class-balanced Random Forest

### Results

  -----------------------------------------------------------------------------
  Model             ROC-AUC       PR-AUC    Precision       Recall           F1
  ------------ ------------ ------------ ------------ ------------ ------------
  Random             0.8469       0.3712       0.5762       0.1810       0.2755
  Forest                                                           

  Random             0.8468       0.3468       0.4266       0.3636       0.3926
  Forest                                                           
  (Balanced)                                                       
  -----------------------------------------------------------------------------

Random Forest provided better discrimination than Logistic Regression,
but did not reach the project's 0.85 ROC-AUC target by much.

------------------------------------------------------------------------

# 4. XGBoost Baseline

XGBoost was trained using:

``` python
XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)
```

### Results

  Metric            Result
  ----------- ------------
  ROC-AUC       **0.8693**
  PR-AUC        **0.4100**
  Precision     **0.6051**
  Recall          \~0.2000
  F1            **0.3018**

The baseline XGBoost model exceeded the project's target of **0.85
ROC-AUC** on the validation set.

------------------------------------------------------------------------

# 5. XGBoost Hyperparameter Tuning

RandomizedSearchCV was used to search across different XGBoost
hyperparameters.

Parameters explored included:

-   `n_estimators`
-   `max_depth`
-   `learning_rate`
-   `subsample`
-   `colsample_bytree`
-   `min_child_weight`
-   `gamma`

The search used:

``` text
n_iter = 20
cv = 3
scoring = roc_auc
random_state = 42
```

### Tuned XGBoost Results

  Metric        Result
  ----------- --------
  ROC-AUC       0.8692
  PR-AUC        0.4095
  Precision     0.6065
  Recall        0.1875
  F1            0.2865
  Accuracy      93.76%

### Observation

The tuned model did not improve over the baseline XGBoost model.

The baseline model achieved slightly higher ROC-AUC and PR-AUC, so the
baseline XGBoost model remained the current candidate.

------------------------------------------------------------------------

# 6. Threshold Analysis

The default classification threshold of `0.50` was not treated as
automatically optimal.

Thresholds from `0.10` to `0.50` were tested to study the
precision-recall trade-off.

The analysis showed:

-   Lower threshold → higher recall but lower precision
-   Higher threshold → higher precision but lower recall

At a threshold of `0.20`:

  Metric         Result
  ----------- ---------
  Precision     \~40.0%
  Recall        \~51.1%
  F1            \~44.9%
  Accuracy      \~91.6%

The F1 score was substantially higher around the lower threshold range.

However, `0.20` was **not finalized as the operational banking
threshold**. The final threshold should be selected using the business
cost of false positives versus false negatives.

------------------------------------------------------------------------

# 7. SHAP Explainability

SHAP was added to explain the predictions made by the XGBoost model.

The following analyses were performed:

1.  Global feature importance
2.  SHAP beeswarm plot
3.  Individual prediction explanation using a waterfall plot

## 7.1 Global Feature Importance

The SHAP feature-importance analysis showed that the model relied most
heavily on:

1.  `RevolvingUtilizationOfUnsecuredLines`
2.  `NumberOfTime30-59DaysPastDueNotWorse`
3.  `NumberOfTimes90DaysLate`
4.  `age`
5.  `NumberOfTime60-89DaysPastDueNotWorse`

Other contributing features included:

-   `NumberOfOpenCreditLinesAndLoans`
-   `DebtRatio`
-   `MonthlyIncome`
-   `NumberRealEstateLoansOrLines`
-   `NumberOfDependents`
-   `IncomeMissing`
-   `DelinquencyAnomaly`

## 7.2 SHAP Beeswarm Analysis

The beeswarm plot showed the direction of feature contributions.

General patterns observed included:

-   Higher revolving utilization generally pushed predictions toward
    higher default probability.
-   Higher delinquency counts pushed predictions toward higher default
    probability.
-   Higher age generally pushed predictions toward lower predicted risk.
-   Higher income generally tended toward lower predicted risk.

SHAP describes the model's behavior and feature contribution; it does
not establish that a feature causes default.

## 7.3 Individual Prediction Explanation

A SHAP waterfall plot was generated for an individual validation
borrower.

For that borrower:

-   Revolving utilization had a strong negative contribution.
-   Age had a negative contribution.
-   No recent delinquency contributed toward lower predicted risk.
-   The number of open credit lines contributed slightly toward higher
    risk.

The model output was explained in SHAP log-odds space and then
interpreted as a predicted probability.

------------------------------------------------------------------------

# 8. MLflow Setup

MLflow 3.16.1 was installed and configured for experiment tracking.

The MLflow experiment was created as:

``` text
Loan Default Prediction
```

The XGBoost baseline experiment was logged with:

### Parameters

``` text
n_estimators = 300
max_depth = 4
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
```

### Metrics

``` text
ROC-AUC   = 0.869346
PR-AUC    = 0.409995
Precision = 0.605105
Recall    = 0.200000
F1        = 0.301760
```

------------------------------------------------------------------------

# 9. Full Model Pipeline Logging

Initially, the raw XGBoost model was logged.

However, for deployment, the preprocessing step also needs to be
preserved.

The complete pipeline contains:

``` text
Raw borrower data
       ↓
SimpleImputer
       ↓
XGBoost
       ↓
Prediction
```

The complete pipeline was successfully logged to MLflow as:

``` text
xgb_full_pipeline
```

Status:

``` text
Ready
```

This is important because the eventual FastAPI backend can use the same
preprocessing and model pipeline rather than duplicating preprocessing
logic separately.

------------------------------------------------------------------------

# 10. Current Project Status

The main ML development stage is substantially complete.

``` text
Project Setup                  ✅
Dataset Selection              ✅
EDA                            ✅
Data Cleaning                  ✅
Train/Validation Split         ✅
Logistic Regression            ✅
Random Forest                  ✅
XGBoost                        ✅
XGBoost Tuning                 ✅
Threshold Analysis             ✅
SHAP Explainability            ✅
MLflow Tracking                ✅
Full Pipeline in MLflow        ✅
```

------------------------------------------------------------------------

# 11. Next Steps

The next major stages are:

``` text
Final untouched test evaluation
          ↓
FastAPI backend
          ↓
PostgreSQL integration
          ↓
Streamlit frontend
          ↓
Evidently monitoring
          ↓
Docker
          ↓
AWS ECS deployment
          ↓
Final integration + testing
          ↓
Final report/documentation
```

The next immediate task is to evaluate the finalized model on the
**untouched test dataset** before moving into backend development.
