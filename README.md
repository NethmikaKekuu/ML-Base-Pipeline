# Customer Churn Prediction

A machine learning project that predicts whether a bank customer will churn using Logistic Regression, Decision Tree, and Random Forest classifiers — with full preprocessing pipelines, cross-validation, hyperparameter tuning, and model evaluation.

---

## 📁 Project Structure

```
├── data/
│   └── processed/
│       ├── Churn_modelling_binning.csv     # Raw input data (binned)
│       └── X_Transformed.csv              # Preprocessed feature data
├── artifacts/
│   ├── X_train.npz                        # Resampled training features (post-SMOTE)
│   ├── Y_train.npz                        # Resampled training labels (post-SMOTE)
│   ├── X_test.npz                         # Test features
│   └── Y_test.npz                         # Test labels
├── notebooks/
│   ├── 01_preprocessing.ipynb             # Data loading, pipelines, SMOTE
│   ├── 02_logistic_regression_basic.ipynb # Basic LR training and evaluation
│   ├── 03_cross_validation.ipynb          # K-Fold CV with best fold selection
│   ├── 04_multi_model.ipynb               # LR vs Decision Tree vs Random Forest
│   ├── 05_hyperparameter_tuning.ipynb     # GridSearchCV tuning
│   └── 06_threshold_tuning.ipynb          # Probability threshold analysis
├── requirements.txt
└── README.md
```

---

## 🔄 Project Workflow

```
Raw CSV Data
     ↓
Preprocessing Pipeline (Imputation + Scaling + Encoding)
     ↓
Handle Class Imbalance with SMOTE
     ↓
Train/Test Split → Save as .npz artifacts
     ↓
Model Training (LR, Decision Tree, Random Forest)
     ↓
Cross Validation (StratifiedKFold) → Best Fold Selection
     ↓
Hyperparameter Tuning (GridSearchCV)
     ↓
Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
     ↓
Probability Threshold Tuning
```

---

## ⚙️ Setup and Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd churn-prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Notebooks Explained

### `01_preprocessing.ipynb` — Data Preparation
- Loads `Churn_modelling_binning.csv`
- Builds three separate sklearn `Pipeline` objects:
  - **Numerical**: Median imputation → StandardScaler
  - **Nominal**: Constant imputation → OneHotEncoder (for Gender, Geography)
  - **Ordinal**: Constant imputation → OrdinalEncoder (for CreditScoreBins)
- Combines them using `ColumnTransformer`
- Handles class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique)
- Saves train/test splits as compressed `.npz` files in `artifacts/`

### `02_logistic_regression_basic.ipynb` — Baseline Model
- Loads preprocessed artifacts
- Trains a `LogisticRegression` model on the full training set
- Generates hard predictions (`predict`) and soft probabilities (`predict_proba`)
- Evaluates with Accuracy, Precision, Recall, F1, and Confusion Matrix

### `03_cross_validation.ipynb` — Cross Validation
- Configures `StratifiedKFold` with 6 splits
- Runs cross validation across 4 metrics: Accuracy, Precision, Recall, F1
- Identifies the best performing fold using `np.argmax`
- Retrains a fresh model on that fold's training data
- Evaluates on the held-out test set

### `04_multi_model.ipynb` — Model Comparison
- Trains and cross-validates three models simultaneously:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Uses `return_estimator=True` to extract the best estimator per model
- Plots a Confusion Matrix for each model side by side

### `05_hyperparameter_tuning.ipynb` — GridSearchCV
- Defines parameter grids for each model:
  - **Logistic Regression**: `max_iter`
  - **Decision Tree**: `max_depth`, `criterion`
  - **Random Forest**: `max_depth`, `n_estimators`, `criterion`
- Runs `GridSearchCV` with `StratifiedKFold` CV and F1 scoring
- Reports best parameters and best CV score per model

### `06_threshold_tuning.ipynb` — Probability Threshold Analysis
- Uses `predict_proba` to get raw probability scores
- Plots the probability distribution of predictions
- Adjusts the classification threshold from default 0.5
- Re-evaluates model with custom threshold to improve Recall or Precision depending on business need

---

## 📊 Features Used

| Feature | Type | Transformer |
|---|---|---|
| Age | Numerical | Median Imputer + StandardScaler |
| Tenure | Numerical | Median Imputer + StandardScaler |
| Balance | Numerical | Median Imputer + StandardScaler |
| EstimatedSalary | Numerical | Median Imputer + StandardScaler |
| Gender | Nominal | Constant Imputer + OneHotEncoder |
| Geography | Nominal | Constant Imputer + OneHotEncoder |
| CreditScoreBins | Ordinal | Constant Imputer + OrdinalEncoder |
| NumOfProducts | Remainder | Passed through as-is |
| HasCrCard | Remainder | Passed through as-is |
| IsActiveMember | Remainder | Passed through as-is |

**Target variable:** `Exited` (1 = churned, 0 = stayed)

---

## 🤖 Models

| Model | Key Parameters Tuned |
|---|---|
| Logistic Regression | `max_iter` |
| Decision Tree | `max_depth`, `criterion` |
| Random Forest | `max_depth`, `n_estimators`, `criterion` |

---

## 📈 Evaluation Metrics

| Metric | What It Measures |
|---|---|
| **Accuracy** | Overall correct predictions out of all predictions |
| **Precision** | When model predicts churn, how often is it right |
| **Recall** | Out of all actual churners, how many did model catch |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Visual breakdown of TP, TN, FP, FN |

> **Note:** F1 Score is used as the primary metric because the dataset has class imbalance — accuracy alone would be misleading.

---

## ⚖️ Handling Class Imbalance

The original dataset has significantly more non-churners than churners. Without handling this, the model would be biased toward predicting "no churn" and achieve high accuracy while being useless.

**Solution: SMOTE (Synthetic Minority Oversampling Technique)**
- Generates synthetic examples of the minority class (churners)
- Balances the training set so the model learns both classes equally
- Applied **only to training data** — test data remains untouched to reflect real-world distribution

---

## 🔑 Key Concepts Used

- **Sklearn Pipelines** — chain preprocessing steps to prevent data leakage
- **ColumnTransformer** — apply different transformations to different feature types
- **StratifiedKFold** — maintain class balance across all CV folds
- **cross_validate** — evaluate model across multiple folds and metrics
- **GridSearchCV** — exhaustively search for best hyperparameters
- **SMOTE** — oversample minority class to fix class imbalance
- **Probability Thresholding** — tune the decision boundary beyond default 0.5

---

## 📦 Requirements

```
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
seaborn
joblib
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 👤 Author

Nethmika Kekulanthale
