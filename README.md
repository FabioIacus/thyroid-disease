# Thyroid Disease Classification (ANN-Thyroid)

This repository contains the code and report for a machine learning assignment on the **ANN-Thyroid** dataset from the UCI Machine Learning Repository.  
The goal is to compare several classical classification algorithms on a real, highly imbalanced medical diagnosis problem.

---

## Project overview

We tackle a **3-class classification** task:

- **Class 1** – Hyperthyroid  
- **Class 2** – Subnormal  
- **Class 3** – Normal  

Main steps:

1. Data loading and cleaning (merging UCI train/test files, handling missing values).
2. Exploratory Data Analysis (EDA) to inspect feature distributions and correlations.
3. Train/validation/test split with **stratification** (60% / 20% / 20%).
4. Model training and hyperparameter tuning via **Stratified 5-fold cross-validation**.
5. Evaluation with metrics robust to class imbalance (macro Precision/Recall/F1, ROC–AUC).
6. Comparative analysis and discussion in the LaTeX report.

---

## Dataset

- **Name:** Thyroid Disease – ANN-Thyroid  
- **Source:** UCI Machine Learning Repository  
- **Link:** <https://archive.ics.uci.edu/dataset/102/thyroid+disease>  
- **Samples (raw):** 7200  
- **Samples (after cleaning):** 7129  
- **Features used:** 21 input variables (mix of binary and continuous)  
- **Task:** Multiclass classification (3 classes)

**Important:**  
The original UCI data files **are not included** in this repository.  
To reproduce the experiments, download:

- `ann-train.data`  
- `ann-test.data`

from the UCI page and place them in the working directory (or update the paths in the notebook).

---

## Models

The following models are implemented and compared (all via scikit-learn):

- **Gaussian Naive Bayes**
- **Softmax regression** (multinomial logistic regression)
- **Decision Tree**
- **Random Forest**
- **Linear SVM**
- **RBF SVM**

For each model we perform:

- Hyperparameter tuning with `GridSearchCV` + `StratifiedKFold` (5 folds)
- Multi-metric scoring: accuracy and **macro F1** (the latter used for `refit`)
- Evaluation on validation and test sets with:
  - Accuracy  
  - Macro Precision  
  - Macro Recall  
  - Macro F1  

We also generate:

- Confusion matrices (validation + test)
- ROC curves and macro ROC–AUC (OvR)
- Learning curves (train vs validation performance)
- Feature importance plots (Decision Tree + permutation importance)
- A 2D decision boundary visualization using PCA + RBF SVM

---

## How to run
Option 1 – Google Colab

1. Upload thyroid_disease.ipynb to Google Colab.

2. Upload ann-train.data and ann-test.data to /content/ (Colab file system),
or mount Google Drive and adjust TRAIN_PATH and TEST_PATH in the notebook.

3. Run all cells from top to bottom.

Option 2 – Local environment

1. Create a Python environment (e.g. via venv or conda).

2. Install the required packages (approximate list):
  pip install numpy pandas matplotlib scikit-learn


3. Put ann-train.data and ann-test.data in the project folder.

4. Open thyroid_disease.ipynb with Jupyter Notebook, JupyterLab, or VS Code and run all cells.

## Results (summary)

- The dataset is strongly imbalanced (≈92.5% of samples belong to the “normal” class).

- Gaussian Naive Bayes performs very poorly because its conditional independence and Gaussian assumptions are strongly violated; it achieves low accuracy and very low macro F1.

- Softmax regression and Linear SVM achieve high accuracy and reasonable macro F1, but still struggle more on the minority classes than the best tree-based models.

- Decision Tree and Random Forest achieve the best macro F1 and almost perfectly diagonal confusion matrices, handling non-linearities and mixed-type features well.

- Random Forest slightly outperforms a single tree but with higher computational cost and lower interpretability.

- Macro-averaged metrics and ROC–AUC are crucial: simple accuracy would be dominated by the majority class and would hide important differences between models.

## License / usage

This repository is intended for educational purposes as part of a Machine Learning course assignment.
The ANN-Thyroid dataset is provided by the UCI Machine Learning Repository and is subject to their terms of use.
