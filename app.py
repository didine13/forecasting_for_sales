# Import
import streamlit as st
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Initial state of website
st.set_page_config(
    page_title="Forecast for sales",
    page_icon="💰",
    layout="wide")
# css style
CSS = """
h1 {
    color: red;
}
.stApp {
    background-color: #2C2E43;
}
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.title('Forecast for sales')

st.header('Management inventory')


# st.dataframe(my_dataframe)
expander_df = st.expander(label='DataFrame')

with expander_df:
    df = pd.read_csv('raw_data/train_all_table.csv', nrows=500000).drop(columns='Unnamed: 0')

    option_head = st.slider('head : ', 1, 100, 5)
    st.write(df.head(option_head))

st.write('Date minimum : ', min(df['date']))
st.write('Date maximum : ', max(df['date']))

# Inventory Units (With + or -) %
# Sales Units (With + or -) %
col1, col2 = st.columns(2)
col1.metric("Inventory Units", "437.8", "-$1.25")
col2.metric("Sales Units", "121.10", "0.46%")

col_show1, col_show2 = st.columns(2)


# Test directly plot
# ------------------

fig , axes = plt.subplots(2, 2, figsize=(15, 16))

sns.countplot(ax=axes[0, 0], y=df['family'])
sns.countplot(ax=axes[0, 1], y=df['type_x'])
sns.countplot(ax=axes[1, 0], y=df['state'])

st.pyplot(fig)

# ------------------


# Checkbox to display something
# Columns 1
with col_show1:
    if st.checkbox('Show 1', value=True):
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


    # Third
    # -----------------------
    if st.checkbox('Show 3'):
        st.write('''
            Screen 3 : Inventory Trend
            ''')
        # check for

        st.write('''
            Screen 3 : Inventory Efficient (Lines predict, Lines real)
            ''')

# Columns 2
with col_show2:
    # Second
    # -----------------------
    if st.checkbox('Show 2', value=True):
        st.write('''
            Screen 2 : Top 10 of sales, invetories
            ''')

        st.write('''
            Screen 2 : Needed product (Prod, Alert, Nb)
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
if st.checkbox('Show Map', value=True):

    st.table(df['city'].unique())

    # from geopy.geocoders import Nominatim

    # for city in df['city'].unique():

    #     address = df['city']
    #     geolocator = Nominatim(user_agent="name")
    #     location = geolocator.geocode(address)
    #     print(location.address)
    #     print((location.latitude, location.longitude))
    #     df[df['city'] == city]['lat'] = location.latitude
    #     df[df['city'] == city]['lon'] = location.longitude


    st.write('''
        Screen Map : with Folium (later)
        ''')

    st.map()

    # search latitude, longitude
    # coordinates of city, state
    # display details of stores when hover with mousepad
