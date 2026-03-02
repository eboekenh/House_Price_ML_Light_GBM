"""
House_price_machine_learning.py
-------------------------------
End-to-end machine learning pipeline for the Kaggle House Prices
competition.  The pipeline covers:
  1. Data loading & merging (train + test)
  2. Exploratory Data Analysis (EDA)
  3. Feature Engineering
  4. Missing value imputation
  5. Outlier handling
  6. Rare-category encoding
  7. Label / One-Hot encoding
  8. Model training, tuning, and evaluation with LightGBM
"""

import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMRegressor

from helpers.data_prep import *
from helpers.eda import *
import matplotlib.pyplot as plt
import missingno as msno

# Suppress non-critical warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

# Pandas display options — show all columns/rows; format floats to 3 decimals
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

######################################
# 1. DATA LOADING
######################################

# Load train and test sets, then concatenate them for joint preprocessing
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
df = train.append(test).reset_index(drop=True)
df.head()
#df.info
df.shape

df.columns

######################################
# 2. EXPLORATORY DATA ANALYSIS (EDA)
######################################

# Quick dataframe overview: shape, types, head/tail, NAs, quantiles
check_df(df)

# Classify columns into categorical, high-cardinality, numerical, and
# numeric-but-categorical groups
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)

######################################
# 2a. Categorical Variable Analysis
######################################

for col in cat_cols:
    cat_summary(df, col)

for col in cat_but_car:
    cat_summary(df, col)

for col in num_but_cat:
    cat_summary(df, col)

######################################
# 2b. Numerical Variable Analysis
######################################

# Descriptive statistics at selected percentiles
df[num_cols].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

# Histogram for each numerical variable
for col in num_cols:
    num_summary(df, col, plot=True)

######################################
# 3. FEATURE ENGINEERING
######################################

# Investigate GarageCars vs SalePrice — box plot shows distinct sale-price
# distributions across garage-car groups, so no category merging is applied.
sns.boxplot(x="GarageCars", y="SalePrice", data=df,
            whis=[0, 100], width=.6, palette="vlag")
plt.show()

# --- Bathroom features ---
# Total bathrooms (half baths count as 0.5)
df["TOTALBATH"] = df.BsmtFullBath + df.BsmtHalfBath*0.5 + df.FullBath + df.HalfBath*0.5
# Full bathrooms (basement + above grade)
df["TOTALFULLBATH"] = df.BsmtFullBath + df.FullBath
# Half bathrooms (basement + above grade)
df["TOTALHALFBATH"] = df.BsmtHalfBath + df.HalfBath

# --- Area features ---
# Total area: basement square footage + above-grade living area
df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']

# Total floor area: first floor + second floor
df['TotalFloorSF'] = df['1stFlrSF'] + df['2ndFlrSF']

# Total porch area: open porch + enclosed porch + three-season porch + screen porch
df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

# --- Date / age features ---
# Convert year columns to datetime objects for date arithmetic
df['YearBuilt'] = pd.to_datetime(df['YearBuilt'], format='%Y')
df['YearRemodAdd'] = pd.to_datetime(df['YearRemodAdd'], format='%Y')
df['GarageYrBlt'] = pd.to_datetime(df['GarageYrBlt'], format='%Y')
df['YrSold'] = pd.to_datetime(df['YrSold'], format='%Y')

# If garage build year is missing, use the main building year
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].isnull(), df['YearBuilt'], df['GarageYrBlt'])

# Reference date — the latest sale date in the dataset (2010-01-01)
df['YrSold'].max()
current_date = pd.to_datetime('2010-01-01 0:0:0')

# Building age in days (relative to the reference date)
df['BuiltAge'] =(current_date - df['YearBuilt']).dt.days

# Days between sale date and construction date
df['Sold-Built'] =(df['YrSold'] - df['YearBuilt']).dt.days

# Days between sale date and last remodel date
df['Sold-RemodeAdd'] =(df['YrSold'] - df['YearRemodAdd']).dt.days

# Drop the original year columns (replaced by engineered features above)
df.drop(["YearBuilt","YearRemodAdd","GarageYrBlt","YrSold"],axis=1,inplace=True)

# Reclassify column types after feature engineering
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)

######################################
# 4. MISSING VALUE HANDLING
######################################

missing_values_table(df)
df.head()

# Garage-related columns: NaN means "no garage" — replace with None
df[['GarageFinish','GarageQual','GarageCond','GarageType']] = df[['GarageFinish','GarageQual','GarageCond','GarageType']].replace(np.NaN,None)
df.isnull().sum()

# Drop columns with very high missing-value percentages
df.drop(["PoolQC","MiscFeature","Alley","Fence"],axis=1,inplace=True)
df.columns

# FireplaceQu: presence/absence impacts SalePrice, so convert to binary flag
df["FireplaceQu"] = df["FireplaceQu"].isnull().astype('int')

df.corr()

# Fill remaining missing numerical values with the column median
na_cols = [col for col in num_cols if df[col].isnull().sum() > 0 and "SalePrice" not in col]
df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0)

df.isnull().sum()

df.GarageCars.dtype

# Fill remaining missing categorical values with the column mode
df = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == "O" else x, axis=0)

missing_values_table(df)
df.head()

# Reclassify column types after imputation
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)


######################################
# 5. OUTLIER HANDLING
######################################

# Check which numerical columns contain outliers
for col in num_cols:
    print(col, check_outlier(df, col))


# Visual outlier inspection via box plots (uses 25th/75th percentile whiskers)
def catch_outlier(dataframe):
    """Display a box plot for every column in the dataframe."""
    for i in dataframe.columns:
        sns.boxplot(x=dataframe[i])
        plt.show()

catch_outlier(df)

# Cap (winsorize) outliers to the 1st/99th-percentile thresholds
for col in num_cols:
   replace_with_thresholds(df,col)

# Verify that no outliers remain after capping
for col in num_cols:
    print(col, check_outlier(df, col))



######################################
# 6. RARE CATEGORY ENCODING
######################################

# Identify rare categories (< 1 % frequency) and merge them into 'Rare'
rare_analyser(df, "SalePrice", 0.01)
df = rare_encoder(df, 0.01)
rare_analyser(df, "SalePrice", 0.01)

######################################
# 7. LABEL ENCODING & ONE-HOT ENCODING
######################################

# Include high-cardinality columns for one-hot encoding as well
cat_cols = cat_cols + cat_but_car

df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)

# Apply label encoding to any remaining binary columns
binary_cols = [col for col in df.columns if len(df[col].unique()) ==2]

for col in binary_cols:
    df = label_encoder(df, col)


######################################
# 8. TRAIN / TEST SPLIT
######################################

# Separate back into train (has SalePrice) and test (SalePrice is NaN)
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

# Drop any remaining rows with missing values in the training set
train_df.dropna(inplace=True)

# Persist preprocessed dataframes for later use
train_df.to_pickle("train_df.pkl")
test_df.to_pickle("test_df.pkl")

#######################################
# 9. MODEL: LightGBM
#######################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)
df.isnull().sum()
train_df.head()
test_df.head()

# Prepare feature matrix X and log-transformed target y
X = train_df.drop(['SalePrice', "Id"], axis=1)
y = np.log1p(train_df['SalePrice'])

# 80/20 train-test split with a fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

# Baseline LightGBM model with default hyperparameters
lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


#######################################
# 10. HYPERPARAMETER TUNING
#######################################

# Define the hyperparameter search space
lgbm_params = {"learning_rate": [0.005, 0.1],
               "n_estimators": [1000,10000],
               "max_depth": [2, 3, 5],
               "colsample_bytree": [0.5, 0.2],
               "num_leaves":[3,5,8,10]}

# Exhaustive grid search with 10-fold cross-validation
lgb_model = LGBMRegressor()
lgb_cv_model = GridSearchCV(lgb_model, lgbm_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
lgb_cv_model.best_params_
'''
Best parameters found:
{'colsample_bytree': 0.2,
 'learning_rate': 0.005,
 'max_depth': 3,
 'n_estimators': 10000,
 'num_leaves': 8}
'''

#######################################
# 11. FINAL MODEL
#######################################

# Refit with the best hyperparameters on the entire feature set
lgbm_tuned = LGBMRegressor(**lgb_cv_model.best_params_).fit(X,y)

# Evaluate on training split
y_pred = lgbm_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# Evaluate on hold-out test split
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Kaggle public leaderboard score: 0.126

#######################################
# 12. FEATURE IMPORTANCE
#######################################

def plot_importance(model, features, num=len(X), save=False):
    """Plot a horizontal bar chart of the top *num* most important features
    according to the trained model."""
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(lgbm_tuned, X_train, 20)
