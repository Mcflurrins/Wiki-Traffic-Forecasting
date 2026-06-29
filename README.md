# Web Traffic Time Series Forecasting with Neural Networks

## Overview

This project tackles the [Kaggle Web Traffic Time Series Forecasting competition](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/leaderboard), predicting daily Wikipedia article traffic for 145k articles across 62 days (September 13 – November 13, 2017).

The solution uses a shallow **multi-input 
neural network** that combines time-series features (weekly median deviations), categorical metadata (Wikipedia language and access type), and learned embeddings to forecast visitor traffic under the SMAPE (Symmetric Mean Percentage Error) metric.

---

## Key Insights & Approach

### 1. **Recency Bias**
Recent data is most predictive of near-future traffic. The model trains on a **1-year rolling window** (March 12 – September 10, 2016) and validates on the same future period (September 13 – November 13, 2016) to simulate the competition setting.

### 2. **Log Transformation & Right-Skewed Data**
Wikipedia traffic is heavily right-skewed (some viral articles get 500k+ views). The solution applies **log1p transformation** to compress the scale and stabilize gradients during neural network training.

### 3. **Per-Series Normalization**
Each article's traffic pattern is distinct. The model normalizes the target by subtracting each article's median visit count (computed in log-space), allowing the network to learn **deviations from typical behavior** rather than absolute values.

### 4. **Feature Engineering**

#### Temporal Features
- **8-week median deviations**: Rolling medians computed over non-overlapping 7-day windows, normalized by subtracting each article's overall median
- Captures both trend and cyclical patterns (e.g., weekends vs. weekdays)

#### Categorical Features
Extracted from the page name format: `name_project_access_agent` (e.g., `AKB48_zh.wikipedia.org_all-access_spider`)
- **Wikipedia project/language** (9 categories): one-hot encoded → embedded in the neural network
- **Access type**: desktop, mobile, all-access
- **Agent type**: human, spider, all-agents

#### Data Imputation
Missing values are forward-filled and any remaining gaps are zero-filled, treating zeros as either missing data *or* legitimate zero traffic.

### 5. **Model Architecture**

A shallow multi-input neural network with three branches:
1. **Main input**: 8 normalized weekly median deviations (fully connected layers)
2. **Site input**: 9-element one-hot vector for Wikipedia language (embedded, then concatenated)
3. **Access input**: one-hot access type (optionally embedded)

All branches merge and feed into fully connected dense layers, outputting a **single predicted visit count** in log-space (to be inverse-transformed).

Activation functions: ReLU hidden layers, linear output (MSE loss).

---

## Project Structure

- **Data Loading**: Unzips and loads train, key, and sample submission CSVs
- **Metadata Extraction**: Parses page names to extract site, access agent, and other categorical features
- **Feature Construction**: 
  - Computes rolling weekly medians
  - Normalizes features per article
  - One-hot encodes categorical variables
- **Model Definition**: Keras/TensorFlow multi-input neural network
- **Training & Validation**: Trains on 2016 data, validates on overlapping future period
- **Submission Generation**: Predicts on the test period and formats output for Kaggle

---

## Data Summary

| Dataset | Shape | Coverage |
|---------|-------|----------|
| **Training** | (145,063 articles, 804 days) | July 1, 2015 – Sept 10, 2017 |
| **Validation Period** | Sept 13 – Nov 13, 2016 | 62 days (competition window) |
| **Key Mapping** | (8,993,906 rows, 2 cols) | Article + date → Submission ID |
| **Submission Format** | (8,993,906 rows, 2 cols) | Submission ID → Predicted visits |

---

## How to Use

### Prerequisites
```bash
python >= 3.8
pandas
numpy
tensorflow / keras
```

### Steps

1. **Download Data**
   - Get `train_2.csv.zip`, `key_2.csv.zip`, and `sample_submission_2.csv.zip` from the [competition page](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/data)
   - Extract or load them in the notebook

2. **Run the Notebook** (`my-project.ipynb`)
   - Load training data and explore structure
   - Extract categorical features from page names
   - Compute rolling weekly medians and normalize features
   - Build the multi-input neural network
   - Train on 2016 data (validation on Sept 13 – Nov 13, 2016)
   - Generate predictions on the test period (Sept 13 – Nov 13, 2017)

3. **Submit**
   - Format predictions as required: ID → Predicted visits
   - Upload to Kaggle

---

## Evaluation Metric

**SMAPE (Symmetric Mean Absolute Percentage Error)**

$$\text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|) / 2}$$

SMAPE is symmetric and handles both high and low traffic articles fairly, penalizing relative errors rather than absolute ones.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **1-year rolling window** | Captures seasonal patterns; trains on most recent year-to-date data for relevance |
| **Log1p transformation** | Stabilizes right-skewed traffic data; reduces gradient explosion |
| **Per-series median normalization** | Allows model to learn deviations from each article's "typical" behavior |
| **Weekly median aggregation** | Smooths noise; captures 7-day cycles (weekday/weekend effects) |
| **Zero-fill imputation** | Treats missing data conservatively; zeros are valid traffic counts |
| **Shallow MLP + embeddings** | Balance between expressiveness and regularization; avoids overfitting on sparse categorical features |

---

## Results & Performance

The model is validated using **SMAPE** on the holdout test period, getting a public and private score of 40.47791 on the private and public leaderboards, around 5 points away from the top scorer. Multi-input architecture with categorical embeddings typically outperforms single-input baselines, as it leverages language-specific and access-type patterns in traffic.

WIP: XGBoost + Lag
This model was designed with interpretability and simplicity in mind. Neural networks are good at catching general patterns, but it'd be worth adding XGBoost with lagged features combined with ratios linked to days of the week or other features to further refine the granular accuracy of our model. 

---

## References

- [Kaggle Competition](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/)
- [1st Place Solution](https://github.com/Arturus/kaggle-web-traffic) – RNN with attention and heavy feature engineering
- SMAPE metric definition: [Wikipedia](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error)
