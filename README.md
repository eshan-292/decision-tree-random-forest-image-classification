# Analysis of ML techniques for Binary and Multi-Class Classification

## Overview
This project is a comprehensive implementation of various machine learning models for binary and multi-class classification. The repository contains implementations from scratch, using `scikit-learn`, and hyperparameter tuning with `GridSearchCV` to optimize performance.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Binary Classification](#binary-classification)
  - [Decision Tree (Scratch Implementation)](#decision-tree-scratch-implementation)
  - [Decision Tree (scikit-learn)](#decision-tree-scikit-learn)
  - [Grid Search and Feature Selection](#grid-search-and-feature-selection)
  - [Cost Complexity Pruning](#cost-complexity-pruning)
  - [Random Forests](#random-forests)
  - [Gradient Boosted Trees & XGBoost](#gradient-boosted-trees--xgboost)
- [Multi-Class Classification](#multi-class-classification)
  - [Decision Tree (scikit-learn)](#decision-tree-multi-class)
  - [Post Pruning and Feature Selection](#post-pruning-and-feature-selection)
  - [Random Forests](#random-forests-multi-class)
  - [Gradient Boosted Trees & XGBoost](#gradient-boosted-trees--xgboost-multi-class)
- [Real-Time Testing](#real-time-testing)
- [Results Summary](#results-summary)
- [Installation](#installation)
- [Usage](#usage)


---

## Binary Classification

### Decision Tree (Scratch Implementation)
- **Time to Build:** 6201.59 seconds
- **Training Metrics:**
  - Accuracy: 98.05%
  - Precision: 96.00%
  - Recall: 96.20%
- **Validation Metrics:**
  - Accuracy: 76%
  - Precision: 54.17%
  - Recall: 26%

### Decision Tree (scikit-learn)
- **Training Time:** 4.32 seconds
- **Training Accuracy:** 98.85%
- **Validation Accuracy:** 93%
- **Best Hyperparameters (Grid Search):**
  - `criterion`: Gini
  - `max_depth`: 5
  - `min_samples_split`: 4

---

### Cost Complexity Pruning
- **Best Alpha:** 0.0027
- **Pruned Tree Validation Accuracy:** 93.5%
- **Validation Precision:** 93.02%

---

### Random Forests
- **Training Accuracy:** 100%
- **Validation Accuracy:** 97%
- **Best Hyperparameters (Grid Search):**
  - `criterion`: Entropy
  - `max_depth`: 7
  - `min_samples_split`: 10
  - `n_estimators`: 100

---

### Gradient Boosted Trees & XGBoost
- **Gradient Boosting Validation Accuracy:** 98%
- **XGBoost Validation Accuracy:** 98.75%
- **Best Hyperparameters for XGBoost:**
  - `max_depth`: 8
  - `n_estimators`: 50
  - `subsample`: 0.6

---

## Multi-Class Classification

### Decision Tree (Multi-Class)
- **Training Accuracy:** 74.25%
- **Validation Accuracy:** 74.25%
- **Best Hyperparameters (Grid Search):**
  - `criterion`: Gini
  - `max_depth`: 7
  - `min_samples_split`: 7

---

### Post Pruning and Feature Selection
- **Best Alpha for Pruning:** 0.0017
- **Pruned Tree Validation Accuracy:** 74.25%

---

### Random Forests (Multi-Class)
- **Validation Accuracy (Default):** 87.25%
- **Validation Accuracy (Optimized):** 89.75%
- **Best Hyperparameters:**
  - `criterion`: Entropy
  - `max_depth`: 10
  - `min_samples_split`: 5
  - `n_estimators`: 200

---

### Gradient Boosted Trees & XGBoost (Multi-Class)
- **Gradient Boosting Validation Accuracy:** 89.25%
- **XGBoost Validation Accuracy:** 90.25%
- **Best Hyperparameters:**
  - `max_depth`: 8
  - `n_estimators`: 50
  - `subsample`: 0.6

---

## Real-Time Testing
- **Dataset:** 10 images (face/dog)
- **Results:** 
  - 7 images predicted as face, 3 as dog
  - **Accuracy:** 70%

---

## Results Summary
The project demonstrated the effectiveness of various models, highlighting the benefits of using ensemble methods like Random Forests and XGBoost, which consistently outperformed basic decision trees.

---

## Installation
To run the project, you need:
- Python 3.x
- Dependencies: `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `xgboost`

Run the following to install:
```bash
pip install -r requirements.txt
