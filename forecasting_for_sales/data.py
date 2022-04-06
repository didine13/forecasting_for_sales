# import
import pandas as pd

def load_big_dataset():
    """Get data from local (raw_data)

    Returns
    -------
    df_train : DataFrame of train

    Notes
    -----

    Version
    -------
    specification : O.S. (v.1 06/04/2022)
    implementation : O.S. (v.1 06/04/2022)
    """
    df_train = pd.read_csv('../raw_data/train.csv')
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
    specification : O.S. (v.1 06/04/2022)
    implementation : O.S. (v.1 06/04/2022)
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    return df

def year_to_csv(df, year):
    """Extract and save in new file.csv, this year's data
    Parameters
    ----------
    df : DataFrame pandas
    year : year of data to keep

    Notes
    -----

    Version
    -------
    specification : O.S. (v.1 06/04/2022)
    implementation : O.S. (v.1 06/04/2022)
    """
    df = df[df['year'] == year]
    df.to_csv(f'../raw_data/train{year}.csv')

def produce_csv_by_year():
    """Produce csv by year (Use previous functions)
    Notes
    -----

    Version
    -------
    specification : O.S. (v.1 06/04/2022)
    implementation : O.S. (v.1 06/04/2022)
    """
    df = load_big_dataset()
    feature_date_engineer(df)
    for i in range(2012, 2018+1):
        year_to_csv(df, i)
