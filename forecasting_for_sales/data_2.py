# import
from ast import arg
import re
from sys import argv
import pandas as pd
import numpy as np
import time

def load_csv(name_file):
    """Load csv
    Parameters
    ----------
    name_file : name of file - String

    Returns
    -------
    pandas.DataFrame : df of your file

    Notes
    -----
    Functions which regroup all load
    functions implemented individually (bad)
    Don't forget to check your folder /raw_data or /data

    Version
    -------
    specification : J.N. (v.1 08/04/2022)
    implementation : O.S. ; J.N. (v.1 08/04/2022)
    """
    print(f'{name_file} has been loaded')
    return pd.read_csv(f'../raw_data/{name_file}.csv')

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

# # Begin for data-preparation_for-preproc

def generate_df_base(df_train):
    """Generate DataFrame which are the base to prepare dataset preproc
    Start='2013-01-01', end='2017-08-15' (train.csv)
    In the end, delete variable df_all_store, df_item
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
    # DataFrame with date - TO DO
    rng = pd.date_range(start='2016-01-01', end='2016-12-31') # remettre 2013_01_01 - 2017_08_15 ------------------
    df_base = pd.DataFrame({'date': rng})

    # Dataframe with combinations store_nbr item_nbr existing in sales dataset
    store_item_combin = df_train[['store_nbr', 'item_nbr']].drop_duplicates()

    #df_base = df_base.merge(df_item, how='cross')
    df_base = df_base.merge(store_item_combin, how='cross')

    # For memory
    del store_item_combin

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
    In the end, delete df_open
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
                        .reset_index()

    # Set True/False in column 'is_open'
    #df_open['is_open'] = (df_open['unit_sales'] != 0)
    df_open['is_open'] = df_open['unit_sales'].apply(lambda x: 1 if x!=0 else 0)

    df_sales = df_sales.merge(df_open[['date', 'store_nbr', 'is_open']],
                              how='left', on=['date', 'store_nbr'])

    # For memory
    del df_open

    return df_sales

def prepare_df_sales(df_train):
    """3 Steps (3functions) : generate df_base -> df_sales which merge with df_open (which prepared)
    Parameters
    ----------
    df_train : DataFrame train

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
    df_base = generate_df_base(df_train)
    df_sales = generate_df_sales(df_base, df_train)
    df_sales = merge_df_open(df_sales)

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
    local_holiday = holiday_data.loc[holiday_data['locale'] == 'Local']
    local_holiday = local_holiday.rename(columns={'locale_name': 'city'})

    # Prepare Regional
    regional_holiday = holiday_data.loc[holiday_data['locale'] == 'Regional']
    regional_holiday = regional_holiday.merge(city_state,
                                              left_on='locale_name',
                                              right_on='state')
    regional_holiday.drop(columns=['state', 'locale_name'], inplace=True)

    # Prepare National
    national_holiday = holiday_data.loc[holiday_data['locale'] == 'National']
    city_state['country'] = 'Ecuador'
    national_holiday = national_holiday.merge(city_state,
                                              left_on='locale_name',
                                              right_on='country')

    # Regroup 3 locales
    df_holiday = pd.concat([local_holiday,
                            regional_holiday,
                            national_holiday])[['date', 'type', 'city']]\
                                .drop_duplicates()
    df_holiday['is_special'] = 1 # all holiday is special

    del local_holiday, regional_holiday, national_holiday, city_state
    print("generated df holidays")
    return df_holiday

def merge_stores(df_sales, stores_data):
    """Merge DataFrame stores to add city, state, ...
    In the end, delete variable store_data
    Parameters
    ----------
    df_sales : DataFrame created with prepare_df_sales()
    stores_data : DataFrame with store_nbr, city, state, ...

    Returns
    -------
    df_sales : DataFrame updated with store content

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
    print("merged stores and sales")
    return df_sales

def merge_df_holiday(df_sales, df_holiday):
    """Generate DataFrame df_holiday to add in df_sales column is_special
    In the end, delete variable df_holiday
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
    # Set to_datetime, important...
    df_holiday['date'] = pd.to_datetime(df_holiday['date'])
    df_sales['date'] = pd.to_datetime(df_sales['date'])
    df_sales.drop(columns=['Unnamed: 0', 'id'], inplace=True)

    stores = pd.read_csv('../raw_data/stores.csv')
    df_sales = df_sales.merge(stores, how='left', on=['store_nbr', 'city', 'state', 'type', 'cluster'])

    df_sales = df_sales.merge(df_holiday, how='left', on=['date', 'city'])
    # Replace NaN by 0
    df_sales['is_special'].fillna(0, inplace=True)

    # For memory
    del df_holiday, stores
    print("merged holidays and sales")
    return df_sales

def merge_items(df_sales, items_data):
    """Merge DataFrame items to add items content
    In the end, delete variable items_data
    Parameters
    ----------
    items_data : DataFrame with store_nbr, city, state, ...

    Returns
    -------
    df_sales : DataFrame updated with items content

    Notes
    -----

    Version
    -------
    specification : J.N. (v.1 08/04/2022)
    implementation : J.N. (v.1 08/04/2022)

    """
    df_sales = df_sales.merge(items_data, how='left', on='item_nbr')

    # For memory
    del items_data
    print("merged items and sales")
    return df_sales

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

def remove_non_perish(df):
    df = df[df['perishable']==1]
    df.drop('perishable', axis=1, inplace=True)
    print("removed non-perishable products and removed perishable column")

if __name__ == '__main__':
    start_time = time.time() #set the beginning of the timer for the execution

    # ----- LOADING MAIN DATASET -----
    #df = load_csv('train')
    df_sales = load_csv('/all_products/train2016') # a remodifier avec le train entier jusqu'à fin 2016 ----------------------------------------
    df_sales = df_optimized(df_sales)

    # ----- PAD WITH DATE AND 0 UNITS WHEN NO UNIT SOLD -----
    df_sales = prepare_df_sales(df_sales)
    df_sales = df_optimized(df_sales)
    print("prepared padded df sales and merged with actual sales")

    # traitement des holidays
    holidays = load_csv('holidays_events_v2') # load holidays
    stores = load_csv('stores') # load stores

    df_holiday = generate_df_holiday(holidays, stores)

    df_holiday.to_csv('../raw_data/df_holiday.csv', index=False)
    # merge sur store
    df_sales = merge_stores(df_sales, stores)

    df_sales.to_csv('../raw_data/df_sales.csv', index=False)

    # merge sur holiday
    df_sales = merge_df_holiday(df_sales, df_holiday)


    items = load_csv('items') # load items
    df_sales = merge_items(df_sales, items) # merge items

    # ----- FEATURE ENGINEER DATE -----
    df_sales = feature_date_engineer(df_sales)
    df_sales.drop(columns=['Unnamed: 0'], inplace=True)
    df_sales = df_optimized(df_sales)

    # ---------------------------------------------------------
    print("merged holidays, items, stores and processed special days")

    # ----- KEEP ONLY PERISHABLE PRODUCTS AND REMOVE PERISHABLE COLUMN -----
    """
    df_sales = remove_non_perish(df_sales)

    family_sales = df_sales.groupby(by=['store_nbr', 'month', 'family'])[['unit_sales']]\
                            .sum()\
                            .reset_index()\
                            .rename(columns={'unit_sales': 'family_sales'})

    family_sales.to_csv('../raw_data/preprocessed_families.csv')
    """
    df_sales.to_csv('../raw_data/preprocessed_sales.csv', index=False)

    print("--- %s seconds ---" % (time.time() - start_time)) #print the timing
