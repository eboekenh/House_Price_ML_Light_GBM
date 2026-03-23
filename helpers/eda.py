"""
eda.py
------
Exploratory Data Analysis (EDA) utilities for the House Price prediction
pipeline.  Provides quick-look summaries for dataframes, column-type
classification, and categorical / numerical / target variable analysis
helpers.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#############################################
# GENERAL
#############################################

def check_df(dataframe):
    """Print a quick overview of the dataframe including shape, data types,
    head/tail rows, missing-value counts, and key quantiles."""
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """Classify dataframe columns into four groups and print a summary.

    Parameters
    ----------
    dataframe : pd.DataFrame
    cat_th : int
        Threshold for the number of unique values below which a numeric
        column is treated as categorical (numeric-but-categorical).
    car_th : int
        Threshold for the number of unique values above which an object
        column is treated as cardinal (categorical-but-cardinal).

    Returns
    -------
    cat_cols : list
        Categorical columns (including numeric-but-categorical).
    cat_but_car : list
        High-cardinality categorical columns.
    num_cols : list
        Numerical columns (excluding numeric-but-categorical).
    num_but_cat : list
        Numeric columns that behave like categorical variables.

    Notes
    -----
    cat_cols + num_cols + cat_but_car covers all columns.
    num_but_cat is already included in cat_cols and is returned only for
    reporting purposes.
    """

    # Object (string) columns
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    # Numeric columns with few unique values → treat as categorical
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    # Object columns with many unique values → high-cardinality
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    # Merge numeric-but-categorical into cat_cols, exclude high-cardinality
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Pure numerical columns (exclude numeric-but-categorical)
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, cat_but_car, num_cols, num_but_cat




#############################################
# CATEGORICAL
#############################################

def cat_summary(dataframe, col_name, plot=False):
    """Print value counts and percentage ratios for a categorical column.
    Optionally display a count plot."""
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


#############################################
# NUMERICAL
#############################################

def num_summary(dataframe, numerical_col, plot=False):
    """Print descriptive statistics at fine-grained quantiles for a
    numerical column.  Optionally display a histogram."""
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()



#############################################
# TARGET
#############################################

def target_summary_with_cat(dataframe, target, categorical_col):
    """Print the mean of the target variable grouped by a categorical column."""
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def target_summary_with_num(dataframe, target, numerical_col):
    """Print the mean of a numerical column grouped by the target variable."""
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")