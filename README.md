# 🏠 House Price Prediction with LightGBM

> **Kaggle competition entry: advanced regression techniques using gradient boosting to predict residential house sale prices.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 📖 About the Project

This project is a solution for the **[Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)** competition. The goal is to predict the final sale price of residential homes in Ames, Iowa, using 79 explanatory variables describing nearly every aspect of the property.

LightGBM (Light Gradient Boosting Machine) was chosen for its speed, efficiency, and strong performance on structured/tabular data with mixed feature types.

---

## 🎯 Problem Statement

**Predict the sale price of a house** given features such as:
- Square footage, lot size, and number of rooms
- Neighborhood and location data
- Construction year and renovation history
- Quality and condition ratings
- Garage, basement, and pool features

---

## ✨ Key Features

- **Feature Engineering**: Handling of missing values, encoding of categorical variables, and creation of new interaction features
- **LightGBM Regression**: Fast gradient-boosted decision tree model optimized for tabular data
- **Cross-Validation**: K-Fold validation to prevent overfitting
- **RMSE Optimization**: Model tuned to minimize Root Mean Squared Log Error (RMSLE)

---

## 🔧 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| LightGBM | Gradient boosting model |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | Preprocessing & cross-validation |
| Matplotlib / Seaborn | EDA visualization |
| Jupyter Notebook | Interactive development |

---

## 📦 Project Structure

```
House_Price_ML_Light_GBM/
│
├── notebooks/
│   └── house_price_lightgbm.ipynb   # Main analysis & modeling notebook
├── data/
│   ├── train.csv                     # Training data (from Kaggle)
│   └── test.csv                      # Test data (from Kaggle)
├── submissions/
│   └── submission.csv                # Kaggle submission file
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/eboekenh/House_Price_ML_Light_GBM.git
cd House_Price_ML_Light_GBM

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebooks/house_price_lightgbm.ipynb
```

> **Note**: Download the dataset from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place `train.csv` and `test.csv` in the `data/` folder.

---

## 📊 Approach

1. **Exploratory Data Analysis** — Understand distributions, outliers, and missing values
2. **Data Preprocessing** — Imputation, encoding, and feature scaling
3. **Feature Engineering** — Create new features from existing ones
4. **Model Training** — LightGBM with K-Fold cross-validation
5. **Hyperparameter Tuning** — Grid/random search for optimal parameters
6. **Submission** — Generate final predictions for Kaggle leaderboard

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**[@eboekenh](https://github.com/eboekenh)**