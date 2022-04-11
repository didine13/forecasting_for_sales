# Import
import streamlit as st
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# Later pip install
# - geopy
# - Nominatim
# - dash

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

'''
# Forecast for sales

## Management inventory
'''


# st.dataframe(my_dataframe)
expander_df = st.expander(label='DataFrame')

with expander_df:
    @st.cache
    def get_cached_data():
        return pd.read_csv('raw_data/train_all_table.csv', nrows=100_000).drop(columns='Unnamed: 0')

    df = get_cached_data()

    option_head = st.slider('head : ', 1, 1000, 5)
    st.write(df.head(option_head))

st.write('Min date: ', min(df['date']))
st.write('Max date: ', max(df['date']))

# Inventory Units (With + or -) %
# Sales Units (With + or -) %
col1, col2 = st.columns(2)
col1.metric("Inventory Units", "437.8", "-$1.25")
col2.metric("Sales Units", "121.10", "0.46%")


if st.checkbox('Show Plot'):
    # Test directly plot
    # ------------------

    fig , axes = plt.subplots(2, 2, figsize=(15, 16))

    sns.countplot(ax=axes[0, 0], data=df, y='family',
                order=df['family'].value_counts().index)\
                    .set_title('Countplot family of products')

    sns.countplot(ax=axes[0, 1], data=df, y='type_x',
                order=df['type_x'].value_counts().index)\
                    .set_title('Countplot type of products')
    sns.countplot(ax=axes[1, 0], data=df, y='state',
                order=df['state'].value_counts().index)\
                    .set_title('Countplot nb products by state')

    sns.histplot(ax=axes[1, 1],data=df, x='unit_sales')\
        .set_title('Countplot onpromotion')

    st.pyplot(fig)


# import plotly.express as px

# # df = px.data.stocks(indexed=True)-1
# fig2 = px.bar(df[df['store_nbr'] == 25], x='date', y='unit_sales')
# fig2.show()

# ------------------

col_show1, col_show2 = st.columns(2)
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

        # Product stock details - begin

        # df[['date', 'items', 'unit_sales']]

        # if unit_sales actual > or <, print 🔻 or ⬆

        st.dataframe(df[['date', 'item_nbr', 'unit_sales']].head(100))


        # Product stock details - end


        # Select box date - min: 2013-01-01 ; max: 2017-08-15 (train)
        start_date = st.date_input('Choose start date', datetime.date(2013, 1, 1))
        end_date = st.date_input('Choose end date', datetime.date(2017, 8, 15))
        st.write('Start date', start_date)
        st.write('End date', end_date)


    # Third
    # -----------------------
    if st.checkbox('Show 3'):
        '''
        Screen 3 : Inventory Trend
        '''
        # Lines plots x=date y=unit_sales
        # fig_inv_trend = px.line(df, x='date', y='unit_sales', markers=True)
        # fig_inv_trend.show()


        '''
        Screen 3 : Inventory Efficient (Lines predict, Lines real)
        '''



# Columns 2
with col_show2:
    # Second
    # -----------------------
    if st.checkbox('Show 2', value=True):
        '''
        Screen 2 : Top 10 of sales, invetories
        '''

        # Hide index of DataFrame
        # CSS to inject contained in a string
        hide_dataframe_row_index = """
                    <style>
                    .row_heading.level0 {display:none}
                    .blank {display:none}
                    </style>
                    """
        # Inject CSS with Markdown
        st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
        df_top_10 = df[['store_nbr', 'unit_sales']].sort_values('unit_sales', ascending=False).head(10)
        st.dataframe(df_top_10)

        '''
        Screen 2 : Needed product (Prod, Alert, Nb)
        '''


    # Fourth
    # -----------------------
    if st.checkbox('Show 4'):
        '''
        Screen 4 : Available stock by departement
        '''

        # days with sliders, textarea
        '''
        Screen 4 : Expire stock within 10 days
        '''
        # st.dataframe()
        # With selectbox of expire days (10, 9, ...)


'''
## Map of stores
'''
# Map
# -----------------------
if st.checkbox('Show Map'):

    st.write(df['city'].unique())

    '''
    Screen Map : with Folium (later)
    '''

    st.map()

    # search latitude, longitude
    # coordinates of city, state
    # display details of stores when hover with mousepad

    # from geopy.geocoders import Nominatim

    # for city in df['city'].unique():

    #     address = df['city']
    #     geolocator = Nominatim(user_agent="name")
    #     location = geolocator.geocode(address)

    #     st.write(address)
    #     st.write(geolocator)
    #     st.write(location)

    #     print(location.address)
    #     print((location.latitude, location.longitude))
    #     df[df['city'] == city]['lat'] = location.latitude
    #     df[df['city'] == city]['lon'] = location.longitude
