# import
from ast import arg
import re
from sys import argv
import pandas as pd
import numpy as np
import time

def load_big_dataset():
    """Get data from local (raw_data)

    Returns
    -------
    df_train : DataFrame of train

    Notes
    -----

    Version
    -------
    specification : J.N. (v.1 06/04/2022)
    implementation : O.S. (v.1 06/04/2022)
    """
    df_train = pd.read_csv('../raw_data/train.csv')
    print("dataset loaded")
    return df_train

def feature_date_engineer(df):
    """Convert date (object) to datetime and add 4 columns (year, month, day, day of week)
    Parameters
    ----------
    df : DataFrame pandas

    Returns
    -------
    df : DataFrame updated with some columns date

    Notes
    -----

    Version
    -------
    specification : J.N. (v.1 06/04/2022)
    implementation : O.S. (v.1 06/04/2022)
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    print("added time features engineered")
    return df

#

def generate_df_base(df_train):
    """Generat DataFrame which are the base to prepare dataset preproc
    Parameters
    ----------
    df_train : DataFrame pandas of train.csv

    Returns
    -------
    df_base : DataFrame with date by store_nbr by item_nbr

    Notes
    -----

    Version
    -------
    specification : J.N. (v.1 08/04/2022)
    implementation : J.N. (v.1 08/04/2022)

    """
    # DataFrame with date
    rng = pd.date_range(start='2016-01-01', end='2016-03-31')
    df_base = pd.DataFrame({'date': rng})

    # DataFrame with store_nbr
    store_nbr_series = df_train['store_nbr'].unique()
    df_all_store = pd.DataFrame({'store_nbr': store_nbr_series})

    df_base = df_base.merge(df_all_store, how='cross')

    # DataFrame with item_nbr
    item_nbr_series = df_train['item_nbr'].unique()
    df_item = pd.DataFrame({'item_nbr': item_nbr_series})

    df_base = df_base.merge(df_item, how='cross')

    # For memory
    del df_all_store, df_item

    return df_base

def generate_df_sales(df_base, df_train):
    """Generate new DataFrame which contains sales
    Parameters
    ----------
    df_base : DataFrame pandas generated with generate_df_base()
    df_train : DataFrame pandas of train.csv

    Returns
    -------
    df_sales : DataFrame updated with unit_sales

    Notes
    -----

    Version
    -------
    specification : J.N. (v.1 08/04/2022)
    implementation : J.N. (v.1 08/04/2022)

    """
    # Set to_datetime, by precaution
    df_base['date'] = pd.to_datetime(df_base['date'])
    df_train['date'] = pd.to_datetime(df_train['date'])

    df_sales = df_base.merge(df_train, how='left', on=['date', 'store_nbr', 'item_nbr'])

    # Replace NaN by 0 (no unit_sales)
    df_sales['unit_sales'] = df_sales['unit_sales'].fillna(0)

    # For memory
    del df_base

    return df_sales

def merge_df_open(df_sales):
    """Update DataFrame which contains if stores open or not
    Parameters
    ----------
    df_sales : DataFrame with date by store_nbr by item_nbr...

    Returns
    -------
    df_sales : DataFrame updated with column is_open

    Notes
    -----

    Version
    -------
    specification : J.N. (v.1 08/04/2022)
    implementation : J.N. (v.1 08/04/2022)

    """
    # keep only 3 columns by ['date', 'store_nbr']
    df_open = df_sales[['date', 'store_nbr', 'unit_sales']]\
                        .groupby(by=['date', 'store_nbr'])\
                        .sum()\
                        .reset_index(inplace=True)

    # Set True/False in column 'is_open'
    df_open['is_open'] = (df_open['unit_sales'] != 0)

    df_sales = df_sales.merge(df_open[['date', 'store_nbr', 'is_open']],
                              how='left', on=['date', 'store_nbr'])

    # For memory
    del df_open

    return df_sales

def generate_df_holiday(holiday_data, stores_data):
    """Generate DataFrame df_holiday to add in df_sales column is_special
    Parameters
    ----------
    holiday_data : DataFrame contains Event, Holiday, Bridge, Work Day, Transfer
    stores_data : DataFrame with store_nbr, city, state, ...

    Returns
    -------
    df_holiday : DataFrame which contains date, type, city, is_special

    Notes
    -----

    Version
    -------
    specification : J.N. (v.1 08/04/2022)
    implementation : J.N. (v.1 08/04/2022)

    """
    # Drop columns not useful
    holiday_data.drop(columns=['description', 'transferred'], inplace=True)
    # Create city_state with stores
    city_state = stores_data[['city', 'state']].drop_duplicates()

    # Prepare Local
    local_holiday = holiday_data[holiday_data.locale == 'Local']
        # new column city in local_holiday, useful for merging
    local_holiday['city'] = local_holiday['locale_name']

    # Prepare Regional
    regional_holiday = holiday_data.loc[holiday_data.locale == 'Regional']
    regional_holiday = regional_holiday.merge(city_state,
                                              left_on='locale_name',
                                              right_on='state')
        # State not useful for merging
    regional_holiday.drop(columns='state', inplace=True)

    # Prepare National
    national_holiday = holiday_data.loc[holiday_data.locale == 'National']
        # Add column country in city_state to merge easily after
    city_state['country'] = 'Ecuador'
    national_holiday = national_holiday.merge(city_state,
                                              left_on='locale_name',
                                              right_on='country')
    # Not useful
    national_holiday.drop(columns=['state', 'country'], inplace=True)

    # Regroup 3 locales
    df_holiday = pd.concat([local_holiday,
                            regional_holiday,
                            national_holiday])[['date', 'type', 'city']]\
                                .drop_duplicates()
    df_holiday['is_special'] = 1 # all holiday is special


    return df_holiday

def merge_stores(stores_data):
    """Generate DataFrame df_holiday to add in df_sales column is_special
    Parameters
    ----------
    stores_data : DataFrame with store_nbr, city, state, ...

    Returns
    -------
    df_sales : DataFrame updated with ..., is_special

    Notes
    -----

    Version
    -------
    specification : J.N. (v.1 08/04/2022)
    implementation : J.N. (v.1 08/04/2022)

    """
    df_sales = df_sales.merge(stores_data, how='left', on='store_nbr')

    # For memory
    del stores_data

    return df_sales



#

def load_items():
    """
    Load items csv
    """
    items = pd.read_csv('../raw_data/items.csv')
    return items

def df_optimized(df, verbose=True, **kwargs):
    """
    Reduces size of dataframe by downcasting numerical columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return:
    """
    in_size = df.memory_usage(index=True).sum()
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df

def remove_outlier(df, replacement_value):
    for item_nbr in df['item_nbr'].unique():
        for store_nbr in df['store_nbr'][df['item_nbr']==item_nbr].unique():
            q1 = df['unit_sales'][(df['item_nbr']==item_nbr) & (df['store_nbr']==store_nbr)].quantile(0.25)
            q3 = df['unit_sales'][(df['item_nbr']==item_nbr) & (df['store_nbr']==store_nbr)].quantile(0.75)
            iqr = q3-q1
            fence_high = q3+1.5*iqr
            df['unit_sales'][(df['item_nbr']==item_nbr) & (df['store_nbr']==store_nbr)].loc[df['unit_sales'] > fence_high] = replacement_value
    print(f"replaced outliers by {replacement_value}")
    return df

def remove_non_perish(df):
    df = df[df['perishable']==1]
    df.drop('perishable', axis=1, inplace=True)
    print("removed non-perishable products and removed perishable column")

def year_to_csv(df, year): # fonction a ne plus utiliser
    """Extract and save in new file.csv, this year's data
    Parameters
    ----------
    df : DataFrame pandas
    year : year of data to keep

    Notes
    -----

    Version
    -------
    specification : J.N. (v.1 06/04/2022)
    implementation : O.S. (v.1 06/04/2022)
    """
    df = df[df['year'] == year]
    df.to_csv(f'../raw_data/train{year}_perish.csv') # adding perish suffix for now, since main function produces csv for perishable products only

def produce_csv_by_year(): # fonction a ne plus utiliser
    """Produce csv by year (Use previous functions)
    Notes
    -----

    Version
    -------
    specification : J.N. (v.1 06/04/2022)
    implementation : O.S. (v.1 06/04/2022)
    """
    df = load_big_dataset()
    feature_date_engineer(df)
    for i in range(2013, 2017+1):
        year_to_csv(df, i)


def produce_csv_by_year_perishable(): # fonction a ne plus utiliser
    """Produce csv by year keeping only perishable products
    Notes
    -----

    Version
    -------
    specification : O.S. (v.1 07/04/2022)
    implementation : O.S. (v.1 07/04/2022)
    """
    df = load_big_dataset()
    feature_date_engineer(df)
    items = load_items()
    df = df.merge(items)
    df = df[df['perishable']==1]
    for i in range(2013, 2017+1):
        year_to_csv(df, i)


if __name__ == '__main__':
    start_time = time.time() #set the beginning of the timer for the execution

    # ----- LOADING MAIN DATASET -----
    df = load_big_dataset()

    # ----- FEATURE ENGINEER DATE -----
    feature_date_engineer(df)


    # ---------------------------------------------------------
    # ----- MERGE HOLIDAYS, STORES AND PROCESS SPECIAL DAYS -----
    # ----- PAD WITH DATE AND 0 UNITS WHEN NO UNIT SOLD -----
    # jonathan
    # merge sur holiday

    # merge sur store

    # traitement des holidays

    # padding par 0

    # ---------------------------------------------------------
    print("merged holidays, stores and processed special days")

    # ----- MERGE ITEMS ON MAIN DATASET -----
    items = load_items()
    df = df.merge(items)
    print("loaded items dataset and merged on main dataset")

    # ----- KEEP ONLY PERISHABLE PRODUCTS AND REMOVE PERISHABLE COLUMN -----
    df = remove_non_perish(df)

    # ----- OPTIMIZE DATA BY DOWNCASTING NUMERIC FEATURES -----
    df = df_optimized(df)

    # ----- CLEAN DATA AND CLEAN ITEMS -----

    # ----- REMOVE NON-SIGNIFICANT STORES, THAT HAVE FEW DATA AVAILABLE -----

    # ----- DROP DUPLICATES (ROWS) -----

    # ----- REPLACING OUTLIER VALUES BY NaN, INTERPOLATE AND SCALE -----
    df = remove_outlier(df, np.nan)

    # ----- DOWNCAST AGAIN ??? -----

    print("--- %s seconds ---" % (time.time() - start_time)) #print the timing
