Formula 1 Race Outcome Prediction
Overview

This project builds a predictive model to estimate the probability that a driver will score points in a Formula 1 race, using historical race, qualifying, and performance data.

The analysis combines race results, qualifying data, and engineered performance and reliability features to evaluate how strongly different factors influence race outcomes.

Objective

The primary goal is to understand:

How predictive starting position is of race outcomes
The extent to which recent driver and constructor performance contributes to results
Whether reliability (DNF rates) materially impacts the likelihood of scoring points
How predictive performance has evolved under modern Formula 1 regulations
Data

Data sourced from Kaggle:

Race results
Qualifying results
Driver and constructor information
Circuit and race metadata
Status (finish vs non-finish)

The dataset spans multiple decades, with modeling focused on modern F1 (2014–present).

Methodology
1. Data Integration
Merged race results with driver, constructor, circuit, and status tables
Constructed a race-level dataset with one row per driver per race
2. Feature Engineering

Created time-aware rolling features using prior race history:

Driver performance:
Average finishing position (last 5 races)
Points scored (last 5 races)
Constructor performance:
Points scored (last 5 races)
Reliability:
Driver DNF rate (last 5 races)
Constructor DNF rate (last 5 races)

All rolling features were calculated using lagged values to prevent data leakage.

3. Modeling Approach

A time-based train/test split was used:

Train: 2014–2021
Test: 2022–present

Models evaluated:

Logistic Regression (baseline and enhanced)
XGBoost (non-linear model)

Performance metric:

ROC-AUC
Results
Model	ROC-AUC
Baseline	~0.85
With Performance Features	~0.87
With Reliability Features	~0.87
XGBoost	~0.85
Key Findings
Starting position (qualifying and grid) is the strongest predictor of race outcomes
Recent driver and constructor performance adds measurable predictive power
Reliability (DNF rates) contributes additional signal, but is secondary to starting position
Predictability remains high in modern F1, suggesting persistent structural advantages
Visualization Highlights
Probability of scoring points by qualifying position
Comparison of actual vs predicted probabilities
SHAP analysis of feature importance
Model lift by prediction decile
Tools & Technologies
Python (pandas, numpy)
scikit-learn
XGBoost
SHAP
matplotlib
Next Steps
Extend model to predict finishing position instead of binary outcome
Incorporate weather and track-specific features
Explore race strategy variables (pit stops, tire compounds)