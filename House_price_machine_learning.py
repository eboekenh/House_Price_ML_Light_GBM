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

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# train ve test setlerinin bir araya getirilmesi.
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
df = train.append(test).reset_index(drop=True)
df.head()
#df.info
df.shape

df.columns


######################################
# EDA
######################################

check_df(df)

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)


######################################
# KATEGORIK DEGISKEN ANALIZI
######################################

for col in cat_cols:
    cat_summary(df, col)

for col in cat_but_car:
    cat_summary(df, col)

for col in num_but_cat:
    cat_summary(df, col)


######################################
# SAYISAL DEGISKEN ANALIZI
######################################

df[num_cols].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

for col in num_cols:
    num_summary(df, col, plot=True)

######################################
# DATA PREPROCESSING & FEATURE ENGINEERING
######################################
#Garage cars feature üretilecek mi üretilmeyecek mi?
sns.boxplot(x="GarageCars", y="SalePrice", data=df,
            whis=[0, 100], width=.6, palette="vlag")
plt.show() # kategoriler birlestirilmedi.Garage carstan feature üretilmedi cünkü gruplarda farkli sales degerleri cikti.


df["TOTALBATH"] = df.BsmtFullBath + df.BsmtHalfBath*0.5 + df.FullBath + df.HalfBath*0.5
df["TOTALFULLBATH"] = df.BsmtFullBath + df.FullBath
df["TOTALHALFBATH"] = df.BsmtHalfBath + df.HalfBath

# Toplam Alan: Bodrum kat yüzölçümü + Garaj Alanı

df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']

#Toplam YüzÖlcümü: Birinci kata ait yüzölçümü,  İkinci kata ait yüzölçümü
df['TotalFloorSF'] = df['1stFlrSF'] + df['2ndFlrSF']

#Balkonların Toplam Yüz Ölçümü
df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

# Yil degiskenini gün cinsinden yazma.Öncesinde yil cinsinden yapmistim ancak degistirip gözlemlemek sitedim

# Yil degiskeninin formatini degistirme
df['YearBuilt'] = pd.to_datetime(df['YearBuilt'], format='%Y')
df['YearRemodAdd'] = pd.to_datetime(df['YearRemodAdd'], format='%Y')
df['GarageYrBlt'] = pd.to_datetime(df['GarageYrBlt'], format='%Y')
df['YrSold'] = pd.to_datetime(df['YrSold'], format='%Y')

# Garaj yapım yılı boşsa bina yapım yılını koyma
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].isnull(), df['YearBuilt'], df['GarageYrBlt'])

# Maksimum Satis Tarihi
df['YrSold'].max() # 01-01-2010 Satış tarihi

current_date = pd.to_datetime('2010-01-01 0:0:0') #referans alınan bugünün tarihi

# Binanin yasi

df['BuiltAge'] =(current_date - df['YearBuilt']).dt.days #bina yaşını veriyor

df['Sold-Built'] =(df['YrSold'] - df['YearBuilt']).dt.days


df['Sold-RemodeAdd'] =(df['YrSold'] - df['YearRemodAdd']).dt.days


#Baslangictaki yil degerlerini atma
df.drop(["YearBuilt","YearRemodAdd","GarageYrBlt","YrSold"],axis=1,inplace=True)
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)


######################################
# MISSING_VALUES
######################################

missing_values_table(df)
df.head()

#Nan'leri Nan ile degistirme (Garaj olmadigi sonucuna varildigi icin)
df[['GarageFinish','GarageQual','GarageCond','GarageType']] = df[['GarageFinish','GarageQual','GarageCond','GarageType']].replace(np.NaN,None)
df.isnull().sum()
#Cok büyük yüzdesi eksik olan degskenleri cikarma
df.drop(["PoolQC","MiscFeature","Alley","Fence"],axis=1,inplace=True)
df.columns

# FireplaceQu degiskeninin varligi ve yoklugu satis degiskeni etkiledigi icin feature'a dönüstürüldü.(Box plot ile gözlenebilir)
df["FireplaceQu"] = df["FireplaceQu"].isnull().astype('int')

df.corr()

# Daha az sayida eksik gözlemi olan sayisal degsikenlere medyan ata
na_cols = [col for col in num_cols if df[col].isnull().sum() > 0 and "SalePrice" not in col]
df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0)

df.isnull().sum()


df.GarageCars.dtype

# Daha az sayida eksik gözlemi olan sayisal degsikenlere mod ata
df = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == "O" else x, axis=0)

missing_values_table(df)
df.head()

#degisken tiplerini tekrar belirle

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df, car_th=10)


######################################
# OUTLIERS
######################################

#outlier var mi diye kontrol etme

for col in num_cols:
    print(col, check_outlier(df, col))


#outlier grafiksel gözlemleme -> ancak box plot %75 %25 aliyor
#sadece hizli bir görsellestirme icin.

def catch_outlier(dataframe):
    for i in dataframe.columns:
        sns.boxplot(x=dataframe[i])
        plt.show()

catch_outlier(df)

#Outlier'lari %99 %1 thresholds ile baskilamaya karar verdim.

for col in num_cols:
   replace_with_thresholds(df,col)

# Outlier var mi yok mu tekrar kontrol etme

#outlier var mi diye kontrol etme

for col in num_cols:
    print(col, check_outlier(df, col))

# outlier'larin baskilandigi görülmekte.



######################################
# RARE ENCODING
######################################

rare_analyser(df, "SalePrice", 0.01)
df = rare_encoder(df, 0.01)

#drop_list = ["Utilities", "LandSlope", "PoolQC", "MiscFeature"]



rare_analyser(df, "SalePrice", 0.01)

######################################
# LABEL ENCODING & ONE-HOT ENCODING
######################################

cat_cols = cat_cols + cat_but_car

df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)

binary_cols = [col for col in df.columns if len(df[col].unique()) ==2]

for col in binary_cols:
    df = label_encoder(df, col)



######################################
# TRAIN TEST'IN AYRILMASI
######################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)
# Kalan az sayida missing gözlemleri düsürme
train_df.dropna(inplace=True)

train_df.to_pickle("train_df.pkl")
test_df.to_pickle("test_df.pkl")

#######################################
# MODEL: LightGBM
#######################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)
df.isnull().sum()
train_df.head()
test_df.head()


X = train_df.drop(['SalePrice', "Id"], axis=1)
y = np.log1p(train_df['SalePrice'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))


#######################################
# Model Tuning
#######################################

lgbm_params = {"learning_rate": [0.005, 0.1],
               "n_estimators": [1000,10000],
               "max_depth": [2, 3, 5],
               "colsample_bytree": [0.5, 0.2],
               "num_leaves":[3,5,8,10]}

lgb_model = LGBMRegressor()
lgb_cv_model = GridSearchCV(lgb_model, lgbm_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
lgb_cv_model.best_params_
'''
{'colsample_bytree': 0.2,
 'learning_rate': 0.005,
 'max_depth': 3,
 'n_estimators': 10000,
 'num_leaves': 8}

'''

#######################################
# Final Model
#######################################

lgbm_tuned = LGBMRegressor(**lgb_cv_model.best_params_).fit(X,y)

y_pred = lgbm_tuned.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

y_pred = lgbm_tuned.predict(X_test)#0.062
np.sqrt(mean_squared_error(y_test, y_pred))#0.059

# Kaggle skoru: 0.126

#######################################
# Feature Importance
#######################################

def plot_importance(model, features, num=len(X), save=False):
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


#######################################
# SONUCLARIN YUKLENMESI
#######################################

submission_df = pd.DataFrame()
submission_df['Id'] = test_df["Id"]

test_df.isnull().sum()
y_pred_sub = lgbm_tuned.predict(test_df.drop("Id", axis=1))
y_pred_sub = np.expm1(y_pred_sub)

submission_df['SalePrice'] = y_pred_sub.astype(int)
submission_df['Id']=submission_df['Id'].astype(int)
submission_df['SalePrice'].dtypes

submission_df.head()
submission_df.dtypes

submission_df.to_csv('submission_lgbm2.csv', index=False)
