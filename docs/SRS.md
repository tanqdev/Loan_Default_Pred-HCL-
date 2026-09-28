# Software Requirements Specification (SRS)

## Loan Default Prediction System

**Domain:** Banking  
**Use-Case:** Bank Muscat

### Technology Stack

**Python 3.13** : Primary programming language used for data processing, machine learning, backend development, and application logic.

**Pandas** : Used for loading, cleaning, transforming, and analyzing structured loan and borrower datasets.

**NumPy** : Used for numerical operations and efficient manipulation of numerical data.

**scikit-learn** : Used for data preprocessing, feature engineering, Logistic Regression, Random Forest, model evaluation, and ML pipelines.

**XGBoost** : Used to build the gradient-boosting model for loan default prediction and improve predictive performance on structured/tabular data.

**SHAP** : Used to explain individual model predictions and identify which borrower/loan features contribute to default risk.

**MLflow** : Used for experiment tracking, logging parameters and metrics, storing model artifacts, and managing model versions.

**FastAPI** : Used to build the real-time REST API through which borrower data is submitted and default-risk predictions are returned.

**PostgreSQL** : Used to store application/prediction data, model-related metadata, timestamps, and other persistent information required by the system.

**Evidently AI** : Used to monitor incoming data and prediction distributions for data drift and identify potential changes that may affect model performance.

**Docker** : Used to containerize the application and its dependencies so the system can run consistently across development and deployment environments.

**AWS ECS** : Used to deploy and run the containerized prediction service in the cloud.

**Matplotlib / Seaborn** : Used for exploratory data analysis and visualization of feature distributions, correlations, class imbalance, and model-related results.

**Jupyter Notebook** : Used during development for exploratory data analysis, experimentation, and visualization.
