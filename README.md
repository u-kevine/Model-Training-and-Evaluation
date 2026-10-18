# Screening for Diabetes and Prediabetes from Self-Reported Health Indicators

**Traditional Machine Learning vs Deep Learning** — Machine Learning Module, Summative Project.

This project predicts whether a person has **prediabetes or diabetes** using only answers to a short
health survey, and compares traditional machine learning against deep learning on the same data.
It is motivated by the fact that many people with diabetes are undiagnosed because diagnosis requires
a lab test; a survey-based pre-screen could direct the right people toward testing.

## Problem framing

- **Target:** binary `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes).
- **Class imbalance:** only 13.9% of respondents are positive, so accuracy is misleading. Models are
  compared on **ROC-AUC and PR-AUC**, and decision thresholds are tuned on validation data, including a
  **screening threshold** chosen for ≥ 80% recall.

## Dataset

CDC Diabetes Health Indicators (UCI Machine Learning Repository, id 891): 253,680 respondents to the CDC's
Behavioral Risk Factor Surveillance System (BRFSS), with 21 health, lifestyle and socio-economic features
and no missing values. Loaded directly with `ucimlrepo` in the notebook (no manual download required).

## Methods

- **Traditional ML (scikit-learn):** Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting,
  Linear SVM and RBF SVM (Nyström), with stratified cross-validation and randomised hyperparameter search.
- **Deep learning (TensorFlow/Keras):** a Sequential MLP explored across 11 experiments (regularisation,
  depth, loss functions including focal loss, optimisers, learning rate and batch size), and a Functional
  **wide-and-deep** model with entity embeddings, both fed via `tf.data` pipelines.
- Clinically motivated feature engineering, a stratified 70/15/15 train/validation/test split, and
  leakage-safe preprocessing fitted on training data only.

## Results (held-out test set, n = 38,052; chance PR-AUC 0.139)

| Model | Test ROC-AUC | Test PR-AUC | Recall @screen |
|---|---|---|---|
| Logistic Regression | 0.821 | 0.403 | 0.794 |
| Random Forest (tuned) | 0.824 | 0.415 | 0.801 |
| HistGradientBoosting (tuned) | 0.827 | 0.418 | 0.795 |
| MLP (Sequential) | 0.828 | 0.420 | 0.801 |
| **Wide & Deep (Functional + embeddings)** | **0.829** | **0.421** | **0.799** |

All competitive models fall within **0.007 ROC-AUC** of each other, and the best traditional and deep
models are statistically indistinguishable (McNemar p = 0.104). Class weighting and focal loss did not
improve ranking but harmed calibration; threshold tuning was the better way to handle imbalance. A model
using only non-clinical questions still reached 0.808 ROC-AUC. Recall was much lower for younger adults
(~42% for ages 18–39 vs ~85% for 60+), a fairness concern for deployment.

## Repository contents

- `diabetes_classification_ml_vs_dl.ipynb` — the full pipeline: loading, EDA, feature engineering,
  modelling, evaluation and error analysis.
- `requirements.txt` — Python dependencies.

## How to run

Open the notebook in Google Colab, run the first cell (`!pip install -q ucimlrepo seaborn`), then select
**Runtime → Run all**. The notebook downloads its own data and saves all figures and result tables
automatically. A fixed random seed (42) is set for reproducibility.

## Author

Kevine Uwisanga
