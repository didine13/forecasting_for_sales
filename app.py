# Import
import streamlit as st
import datetime
import numpy as np
import pandas as pd

st.title('Forecast for sales')

st.header('Management inventory')


# st.dataframe(my_dataframe)
if st.checkbox('Show DataFrame'):
    @st.cache
    def get_dataframe_data():

        # add DataFrame of train_all_table, or others... i don't know
        return pd.DataFrame(
                np.random.randn(10, 5),
                columns=('col %d' % i for i in range(5))
            )

    df = get_dataframe_data()

    st.write(df.head())

# Columns to organize website
columns = st.columns(4)

first_name = columns[0].text_input("First name", value="John")
columns[0].write(first_name)

last_name = columns[1].text_input("Last name", value="Doe")
columns[1].write(last_name)

location = columns[2].text_input("Location", value="Paris")
columns[2].write(location)


# Checkbox to display something
if st.checkbox('Show 1'):
    st.write('''
        Screen 1 : Inventory days of supply
        ''')
    st.write('''
        Screen 1 : Product stock details
        ''')


    # Select box date - min: 2013-01-01 ; max: 2017-08-15 (train)
    start_date = st.date_input('Choose start date', datetime.date(2013, 1, 1))
    end_date = st.date_input('Choose end date', datetime.date(2017, 8, 15))
    st.write('Start date', start_date)
    st.write('End date', end_date)


# Second
# -----------------------
if st.checkbox('Show 2'):
    st.write('''
        Screen 2 : Top 10 of sales, invetories
        ''')

    st.write('''
        Screen 2 : Needed product (Prod, Alert, Nb)
        ''')

    # Inventory Units (With + or -) %
    # Sales Units (With + or -) %


# Third
# -----------------------
if st.checkbox('Show 3'):
    st.write('''
        Screen 3 : Inventory Trend
        ''')

    st.write('''
        Screen 3 : Inventory Efficient (Lines predict, Lines real)
        ''')




# Fourth
# -----------------------
if st.checkbox('Show 4'):
    st.write('''
        Screen 4 : Available stock by departement
        ''')

    # days with sliders, textarea
    st.write('''
        Screen 4 : Expire stock within 10 days
        ''')
    # With selectbox of expire days (10, 9, ...)


# Map
# -----------------------
if st.checkbox('Show Map'):
    st.write('''
        Screen Map : with Folium
        ''')

    # search latitude, longitude
    # coordinates of city, state
    # display details of stores when hover with mousepad



