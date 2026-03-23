# 🏠 House Price Prediction with LightGBM

Regression model predicting residential home sale prices in Ames, Iowa using LightGBM with extensive feature engineering.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-brightgreen)
![Kaggle](https://img.shields.io/badge/Kaggle-0.126_RMSLE-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## Problem Statement

Predict the final sale price of homes based on 80+ features describing physical attributes, location, and condition. Based on the [Kaggle House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Approach

1. **Data Merging** — Train and test sets concatenated for consistent preprocessing
2. **EDA** — Column classification, distribution analysis, correlation inspection
3. **Feature Engineering** — Total bathrooms, total area, porch area, building age, sale-to-build gap
4. **Missing Value Handling** — Garage NaN → "No garage"; high-missingness columns dropped; numerical median / categorical mode fill
5. **Outlier Handling** — IQR-based winsorization (1st/99th percentile)
6. **Rare Category Encoding** — Categories with <1% frequency merged into "Rare"
7. **Encoding** — One-hot encoding for categorical columns, label encoding for binary columns
8. **Model** — LightGBM with GridSearchCV (10-fold CV)

## Results

| Metric | Value |
|--------|-------|
| Kaggle Public Leaderboard (RMSLE) | **0.126** |
| Evaluation Metric | RMSE on log-transformed SalePrice |

**Best Hyperparameters (GridSearchCV):**

| Parameter | Value |
|-----------|-------|
| learning_rate | 0.005 |
| n_estimators | 10,000 |
| max_depth | 3 |
| colsample_bytree | 0.2 |
| num_leaves | 8 |

## Engineered Features

| Feature | Description |
|---------|-------------|
| `TOTALBATH` | Total bathrooms (full + 0.5 × half) |
| `TotalSF` | Basement + above-grade living area |
| `TotalFloorSF` | 1st floor + 2nd floor area |
| `TotalPorchSF` | All porch types combined |
| `BuiltAge` | Building age in days |
| `Sold-Built` | Days between sale and construction |
| `Sold-RemodeAdd` | Days between sale and last remodel |

## Tech Stack

- **Python 3.8+** — Core language
- **LightGBM** — Gradient boosting regressor
- **Pandas / NumPy** — Data manipulation
- **Scikit-learn** — GridSearchCV, metrics, preprocessing
- **Seaborn / Matplotlib** — Visualization

## Project Structure

```
House_Price_ML_Light_GBM/
├── helpers/
│   ├── __init__.py
│   ├── data_prep.py          # Outlier handling, imputation, encoding utilities
│   └── eda.py                # EDA summary and visualization functions
├── House_price_machine_learning.py  # Main ML pipeline
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/eboekenh/House_Price_ML_Light_GBM.git
cd House_Price_ML_Light_GBM
pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from the [Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place them in the project root.

```bash
python House_price_machine_learning.py
```

## License

MIT
