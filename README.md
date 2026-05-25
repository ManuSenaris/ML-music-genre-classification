# Machine-Learning-Deep-Learning-and-AI

# Music Genre Classification

Multi-class supervised classification of songs into 6 musical genres using ensemble tree-based models.  
Developed as part of the **Digital Technology** program at Universidad Torcuato Di Tella (UTDT).

---

## Problem Statement

Given a set of audio features extracted from a song, predict which of the following 6 genres it belongs to:

`acoustic` · `classical` · `electronic` · `hip-hop` · `jazz` · `rock`

The dataset is perfectly balanced across all 6 classes, which validates the use of **F1-Macro** as the primary evaluation metric and eliminates the need for balancing techniques such as SMOTE or class weight adjustment.

---

## Approach

The modeling strategy consisted of training and comparing two ensemble tree-based models:

### 1. Random Forest
Used as the initial model for its interpretability. Feature importance analysis (via Gini impurity reduction) provided insights into which audio features drive genre separation — for example, `danceability` and `speechiness` proved strong predictors for hip-hop.

Hyperparameter optimization was performed using **RandomizedSearchCV** with **4-fold KFold cross-validation** and F1-Macro as the objective metric (`n_iter=100`).

| Hyperparameter | Search Range | Optimal Value |
|---|---|---|
| n_estimators | 50 – 499 | 168 |
| criterion | gini, entropy, log_loss | log_loss |
| max_depth | 5 – 40 | 24 |
| min_samples_split | 2 – 20 | 19 |
| min_samples_leaf | 1 – 15 | 2 |
| min_impurity_decrease | 0.0 – 0.1 | 0.000528 |

### 2. XGBoost
Trained as the second model to improve predictive performance through sequential boosting. Unlike Random Forest, XGBoost required label encoding of the target variable (genres → integers 0–5).

A key finding: optimizing RandomizedSearchCV with **MAP@3** instead of F1-Macro yielded a better Kaggle leaderboard score (0.82911 vs 0.81741), suggesting that probability-based metrics can improve generalization even when the final evaluation metric is different.

**StratifiedKFold** was used to ensure each fold preserves the original class distribution.

| Hyperparameter | Search Range | Optimal Value |
|---|---|---|
| n_estimators | 300 – 799 | 706 |
| learning_rate | 0.03 – 0.23 | 0.051 |
| max_depth | 4 – 9 | 4 |
| min_child_weight | 1 – 5 | 3 |
| subsample | 0.7 – 1.0 | 0.823 |
| colsample_bytree | 0.7 – 1.0 | 0.769 |
| gamma | 0.0 – 3.0 | 0.232 |

---

## Results

| Model | F1-Macro (test) | F1-Macro (CV) | MAP@3 |
|---|---|---|---|
| Random Forest | 0.7818 | 0.7792 | 0.8638 |
| **XGBoost** | **0.8643** | **0.8351** | **0.9119** |

XGBoost outperforms Random Forest across all metrics. The low CV standard deviation (0.0096 and 0.0113 respectively) confirms model stability across folds and the absence of severe overfitting.

**Selected model for submission: XGBoost** — Kaggle score: **0.82911**

---

## Technologies

- **Language:** Python
- **Libraries:** Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## Project Structure

```
├── notebook.ipynb       # Full pipeline: EDA, training, evaluation, submission
├── data/
│   ├── train.csv
│   └── test.csv
└── submission.csv       # Final predictions submitted to Kaggle
```

---

## Key Takeaways

- A perfectly balanced dataset allows F1-Macro to be used as a reliable metric without adjustments.
- Tree-based models are inherently robust to outliers due to their binary split structure.
- The choice of optimization metric in cross-validation directly impacts test generalization — optimizing with MAP@3 produced better Kaggle results than optimizing with F1-Macro directly.
- Low learning rate + high number of estimators in XGBoost is a well-established practice that consistently improves generalization.
