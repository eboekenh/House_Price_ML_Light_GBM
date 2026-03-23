"""
data_prep.py
------------
Data preparation utilities for the House Price prediction pipeline.
Includes functions for outlier detection/handling, missing value analysis,
encoding (label & one-hot), and rare-category encoding.
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing


# =====================================================================
# Outlier Detection & Handling
# =====================================================================

def outlier_thresholds(dataframe, col_name):
    """Calculate lower and upper outlier thresholds using the 1st and 99th
    percentiles and 1.5 × IQR rule."""
    quartile1 = dataframe[col_name].quantile(0.01)
    quartile3 = dataframe[col_name].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """Return True if any value in *col_name* falls outside the outlier
    thresholds, False otherwise."""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_outliers(dataframe, col_name, index=False):
    """Print outlier rows for *col_name*. If more than 10 outliers exist,
    only the first five are printed. Optionally return their index."""
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_outlier(dataframe, col_name):
    """Return a copy of *dataframe* with rows containing outliers in
    *col_name* removed."""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def replace_with_thresholds(dataframe, col_name):
    """Cap (winsorize) outliers in *col_name* to the threshold limits.
    When the lower limit is positive, both lower and upper bounds are applied;
    otherwise only the upper bound is enforced."""
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


# =====================================================================
# Missing Value Analysis
# =====================================================================

def missing_values_table(dataframe, na_name=False):
    """Print a summary table of columns with missing values, showing the
    count and percentage of missing entries. Optionally return the column
    names that contain missing values."""
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    """For each column in *na_columns*, create a binary flag indicating
    missingness and print the target variable's mean and count grouped
    by that flag."""
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


# =====================================================================
# Encoding
# =====================================================================

def label_encoder(dataframe, binary_col):
    """Apply sklearn LabelEncoder to a binary (two-class) column."""
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """Apply one-hot (dummy) encoding to the specified categorical columns."""
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# =====================================================================
# Rare Category Encoding
# =====================================================================

def rare_analyser(dataframe, target, rare_perc):
    """Print category counts, ratios, and target means for categorical
    columns that contain categories appearing less than *rare_perc*
    proportion of the time."""
    rare_columns = [col for col in dataframe.columns if dataframe[col].dtypes == 'O'
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]

    for col in rare_columns:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    """Replace rare categories (those with frequency below *rare_perc*)
    with the label 'Rare' and return the modified dataframe."""
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df