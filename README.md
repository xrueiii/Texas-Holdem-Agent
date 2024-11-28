# FAI Final Project: Texas Hold'em Casino Agent

This project documents the process of implementing and evaluating various machine learning models to create a competitive agent for Texas Hold'em poker. The goal is to develop a machine learning model that maximizes the agent's winnings by making optimal decisions during the game.

---

## Table of Contents
- [Introduction](#introduction)
- [Methods](#methods)
  - [If-Else Determination](#if-else-determination)
  - [Random Forest Classifier](#random-forest-classifier)
  - [Gradient Boosting Classifier (GBDTs)](#gradient-boosting-classifier-gdbts)
- [Configurations](#configurations)
- [Results](#results)
- [Discussion and Conclusion](#discussion-and-conclusion)
- [Future Work](#future-work)

---

## Introduction

Texas Hold'em is a poker game where players combine **hole cards** with **community cards** to form the strongest five-card hand. The agent developed in this project aims to determine optimal actions (`call`, `raise`, `fold`) using machine learning techniques, including ensemble learning methods.

---

## Methods

### If-Else Determination
A rule-based model where:
- **Hand strength** is calculated based on card ranks and suits.
- Actions (`raise`, `call`, `fold`) are determined by specific thresholds.

### Random Forest Classifier
An ensemble learning model trained on features like:
- **Hand strength**
- **Community cards**
- **Opponent behavior**
- **Raise and call amounts**

The agent improves decisions over time by updating the model with new data after each round.

### Gradient Boosting Classifier (GBDTs)
This model replaces Random Forest for its advantages:
- **Higher predictive accuracy**
- **Better handling of class imbalances**
- **Reduced overfitting**

---

## Configurations

### Hyperparameters
#### Random Forest Classifier
- Number of estimators: 80
- Max depth of trees: 5
- Random state: 0

#### Gradient Boosting Classifier
- Number of estimators: 80
- Max depth of trees: 5
- Random state: 0

### Data Preparation
- **Training data** collected during gameplay.
- Split into **80:20** training/testing sets.
- Features include:
  - Hand strength
  - Community cards count
  - Raise/call amounts
  - Opponent behavior
  - Player's stack size

---

## Results
### Performance by Model
- **If-Else Determination**: Limited success, struggled with stronger opponents.
- **Random Forest Classifier**: Improved performance but struggled in complex scenarios.
- **Gradient Boosting Classifier (GBDTs)**: Best performance, high winning rates across all baselines.

---

## Discussion and Conclusion

### Observations
- **GBDTs** outperformed other models in predictive accuracy and adaptability.
- Challenges included:
  - **Data sufficiency**
  - **Computational resources**
  - **Complex opponent behaviors**

### Selected Model
The **Gradient Boosting Classifier (GBDTs)** was selected for its:
- Superior accuracy
- Effective handling of class imbalances
- Reduced overfitting

---

## Future Work

### Exploring XGBoost
- **Efficiency and Speed**: Faster training without compromising accuracy.
- **Enhanced Regularization**: Improved control over overfitting.
- **Scalability**: Better handling of larger datasets and more complex models.

---

## Author
B11705043 - 古昭璿

For more information, visit my GitHub: [xrueiii](https://github.com/xrueiii)
