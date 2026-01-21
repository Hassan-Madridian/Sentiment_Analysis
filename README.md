# Sentiment Analysis on Amazon Reviews (NLP)

## Overview
This project builds an end-to-end **sentiment analysis** system for Amazon product reviews.  
It trains a baseline NLP model using **TF-IDF + Logistic Regression** to classify reviews as **Positive** or **Negative**, and supports predicting sentiment for new/unseen reviews.

---

## Business Problem
Online stores receive thousands of reviews daily. Automatically detecting sentiment helps:
- monitor customer satisfaction
- catch negative feedback early
- summarize product perception at scale

---

## Dataset
Amazon Reviews dataset with 3 columns per row:

1. **label** (typically `1 = negative`, `2 = positive`)
2. **title** (short summary of review)
3. **text** (full review body)

> The dataset is provided as separate `train.csv` and `test.csv` files (no headers).

---

## Approach
### 1) Data Preparation
- Read CSV without headers and assign column names: `label`, `title`, `text`
- Combine `title + text` into `full_text` (improves accuracy by using both signals)
- Map labels to binary:
  - `1 → 0 (Negative)`
  - `2 → 1 (Positive)`

### 2) Model (Baseline)
- **TF-IDF Vectorization**
  - stopword removal (`stop_words="english"`)
  - unigrams + bigrams (`ngram_range=(1,2)`)
  - cap feature size (`max_features=50000`)
- **Logistic Regression**
  - `class_weight="balanced"` to handle class imbalance
  - `max_iter=5000` to ensure convergence

### 3) Evaluation
- Confusion Matrix
- Precision / Recall / F1-score
- Accuracy

---

## Project Structure
