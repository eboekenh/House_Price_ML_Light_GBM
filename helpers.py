"""
helpers.py
----------
General-purpose helper utilities used across various ML projects.
Includes dataframe inspection, outlier handling, CRM data preparation,
invoice-product matrix creation, and CLTV prediction helpers.
"""

from mlxtend.frequent_patterns import apriori, association_rules


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


def outlier_thresholds(dataframe, variable):
    """Calculate lower and upper outlier thresholds using the 1st and 99th
    percentiles and the 1.5 × IQR rule."""
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    """Cap values in *variable* that exceed the upper outlier threshold."""
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def crm_data_prep(dataframe):
    """Prepare a CRM / retail transaction dataframe by removing nulls,
    cancelled invoices, zero-quantity rows, capping outliers, and
    computing a TotalPrice column."""
    dataframe.dropna(axis=0, inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    return dataframe


def create_invoice_product_df(dataframe):
    """Pivot the transaction dataframe into a binary invoice–product matrix
    (1 = product purchased in that invoice, 0 = not purchased)."""
    return dataframe.groupby(['Invoice', 'StockCode'])['Quantity'].sum().unstack().fillna(0). \
        applymap(lambda x: 1 if x > 0 else 0)

import datetime as dt
def create_cltv_p(dataframe):
    """Compute probabilistic Customer Lifetime Value (CLTV) using the
    BG/NBD and Gamma-Gamma models.

    Steps:
      1. Build an RFM table (recency, frequency, monetary).
      2. Fit BG/NBD model to predict expected purchases.
      3. Fit Gamma-Gamma model to predict expected average profit.
      4. Combine into a 6-month CLTV prediction.
      5. Scale CLTV to [1, 100] and assign segments (A/B/C).
    """
    today_date = dt.datetime(2011, 12, 11)

    # Build RFM table — recency is customer-specific (dynamic)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max()-date.min()).days,
                                                                lambda date: (today_date - date.min()).days],
                                                'Invoice': lambda num: num.nunique(),
                                                'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    rfm.columns = rfm.columns.droplevel(0)

    # Rename columns for the CLTV probabilistic model
    rfm.columns = ['recency_cltv_p', 'T', 'frequency', 'monetary']

    # Simplified average monetary value per transaction
    rfm["monetary"] = rfm["monetary"] / rfm["frequency"]

    rfm.rename(columns={"monetary": "monetary_avg"}, inplace=True)


    # Convert recency and customer age (T) to weekly units for BG/NBD
    rfm["recency_weekly_cltv_p"] = rfm["recency_cltv_p"] / 7
    rfm["T_weekly"] = rfm["T"] / 7

    # Filter out customers with non-positive monetary or single purchase
    rfm = rfm[rfm["monetary_avg"] > 0]
    rfm = rfm[(rfm['frequency'] > 1)]

    rfm["frequency"] = rfm["frequency"].astype(int)

    # Fit the BG/NBD model for purchase-frequency prediction
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(rfm['frequency'],
            rfm['recency_weekly_cltv_p'],
            rfm['T_weekly'])

    # Expected number of purchases in the next 1 month (4 weeks)
    rfm["exp_sales_1_month"] = bgf.predict(4,
                                           rfm['frequency'],
                                           rfm['recency_weekly_cltv_p'],
                                           rfm['T_weekly'])
    # Expected number of purchases in the next 3 months (12 weeks)
    rfm["exp_sales_3_month"] = bgf.predict(12,
                                           rfm['frequency'],
                                           rfm['recency_weekly_cltv_p'],
                                           rfm['T_weekly'])

    # Fit the Gamma-Gamma model for expected average profit
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(rfm['frequency'], rfm['monetary_avg'])
    rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm['frequency'],
                                                                             rfm['monetary_avg'])
    # 6-month CLTV prediction
    cltv = ggf.customer_lifetime_value(bgf,
                                       rfm['frequency'],
                                       rfm['recency_weekly_cltv_p'],
                                       rfm['T_weekly'],
                                       rfm['monetary_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)

    rfm["cltv_p"] = cltv

    # Scale CLTV to [1, 100] for easier interpretation
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(rfm[["cltv_p"]])
    rfm["cltv_p"] = scaler.transform(rfm[["cltv_p"]])

    # Segment customers into three CLTV tiers: A (high), B (mid), C (low)
    rfm["cltv_p_segment"] = pd.qcut(rfm["cltv_p"], 3, labels=["C", "B", "A"])

    # Select final columns for output
    rfm = rfm[["recency_cltv_p", "T", "monetary_avg", "recency_weekly_cltv_p", "T_weekly",
               "exp_sales_1_month", "exp_sales_3_month", "expected_average_profit",
               "cltv_p", "cltv_p_segment"]]


    return rfm
