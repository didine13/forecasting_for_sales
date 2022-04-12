# Import
from matplotlib.font_manager import get_fontconfig_fonts
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


expander_df = st.expander(label='DataFrame')

with expander_df:
    @st.cache
    def get_cached_data():
        # return pd.read_csv('raw_data/train_all_table.csv', nrows=10000).drop(columns='Unnamed: 0')
        # work with 1 store
        return pd.read_csv('raw_data/preprocessed_sales_grouped_21.csv')

    df = get_cached_data()
    # df = df[df['date'].dt.year == 2017]


    option_head = st.slider('head : ', 1, 1000, 5)
    st.write(df.head(option_head))

st.write('Min date: ', min(df['date']))
st.write('Max date: ', max(df['date']))


# Inventory Units (With + or -) %
# Sales Units (With + or -) %
col1, col2 = st.columns(2)
col1.metric("Inventory Units", "437.8", "-$1.25")

# show family_sales for this month
# compare previous months
# df['family_sales']
# df.loc[df['date'].dt.year == ]

col2.metric("Sales Units", "121.10", "0.46%")



with st.sidebar:
    '''
    # Forecast for sales

    ## Management inventory
    '''
    '------------------------'
    '''
    - Inventory days of supply
    - Product stock details
    '''
    show_1 = st.checkbox('Show 1')

    '''
    - Top 10 of sales, inventories
    - Needed product (Prod, Alert, Nb)
    '''
    show_2 = st.checkbox('Show 2')

    '''
    - Inventory Trend
    - Inventory Efficient (Lines predict/Real)
    '''
    show_3 = st.checkbox('Show 3')

    '''
    - Available stock by Family
    - Expire stock within 10 days
    '''
    show_4 = st.checkbox('Show 4')

    '''
    Mapping
    '''
    mapping = st.checkbox('Show Map')

# if show_plot:
#     # Test directly plot
#     # ------------------
#     @st.cache
#     def countplot_temp():
#         # fig , axes = plt.subplots(2, 2, figsize=(15, 16))
#         fig , axes = plt.subplots(1, 2, figsize=(15, 16))

#         sns.countplot(ax=axes[0], data=df, y='family_sales',
#                     order=df['family'].value_counts().index)\
#                         .set_title('Countplot family of products')

#         return fig

#     st.pyplot(countplot_temp())



col_show1, col_show2 = st.columns(2)
with col_show1: # Columns 1
    if show_1: # InvDaysSupply, ProdStockDetails
        '''
        ### Screen 1 : Inventory days of supply
        '''
        # Plot with confidence interval - start

        st.write('Something here, a Plot with confidence interval')

        # Plot with confidence interval - end

        # -------------
        # No need maybe
        # -------------

        # '''
        # ### Screen 1 : Product stock details
        # '''
        # # df[['date', 'items', 'family_sales']]
        # # [product, date, unit_hand, unit_order]
        # # if family_sales actual > or <, print 🔻 or ⬆

        # df_stock_details = df[['family', 'date', 'family_sales']]\
        #                     .groupby(by='family').sum()

        # # df_stock_details['date'] = pd.to_datetime(df_stock_details['date'])

        # # if df_stock_details['date'] ==

        # st.dataframe(df_stock_details)

        # -------------
        # No need maybe
        # -------------


    if show_3: # InvTrend, InvEff (Pred, Real)
        '''
        ### Screen 3 : Inventory Trend
        '''
        # Selectbox for year and family
        sb_year_inv = st.selectbox('Year inv',
                        range(min(df['date'].dt.year),
                            max(df['date'].dt.year)))
        sb_family_inv = st.selectbox('Family inv', df['family'].unique())

        # Lines plots x=date y=family_sales by year by family
        @st.cache(suppress_st_warning=True)
        def display_time_series(df, sb_family_inv, year):
            # df = px.data.stocks() # replace with your own data source

            df_family = df.loc[df['family'] == sb_family_inv]\
                            .loc[df['date'].dt.year == year]
            fig = px.line(df_family, x='date', y='family_sales', markers=True)
            return fig

        st.plotly_chart(display_time_series(df, sb_family_inv, sb_year_inv))



with col_show2: # Column 2
    if show_2: # top10, NeedProduct
        '''
        ### Screen right : Top 10 of sales, inventories
        '''
        # add date to choose
        sb_year_top_10 = st.selectbox('Year',
                                range(min(df['date'].dt.year),
                                    max(df['date'].dt.year)))

        def show_top_10(df, sb_year_top_10):
            # df['date'] = pd.to_datetime(df['date'])
            df_in_date = df[df['date'].dt.year == sb_year_top_10]
            df_top_10 = df_in_date[['family', 'family_sales']].groupby(by='family')\
                                .sum()\
                                .sort_values('family_sales', ascending=False).head(10)
            return df_top_10


        st.dataframe(show_top_10(df, sb_year_top_10))

        '''
        ### Screen right : Needed product (Prod, Alert, Nb)
        '''

        def _color_red_or_green(val):
            color = 'red' if val < 10 else 'green'
            return 'color: %s' % color

        # Select store
        option = st.selectbox('Select a line to filter', df['store_nbr'].unique())
        df_store = df[df['store_nbr'] == option]

        df_store['alert'] = (df_store['family_sales'] < 10)

        # add colors
        # df_store.style.apply(_color_red_or_green,
        #                      subset='family_sales', axis=1)
        # By store
        st.dataframe(df_store[['family', 'alert', 'family_sales']])


    if show_4: # AvailableStock, ExpireStock
        '''
        ### Screen 4 : Available stock by family
        '''

        st.write('Add someting here, need item_nbr')
        # Need item_nbr...
        # sb_family_stock = st.selectbox('Family', df['family'].unique())

        # df_family = df[df['family'] == sb_family_stock]
        # df_family = df[df['family'] == sb_family_stock]
        # st.plotly_chart(px.bar(df_family,x='unit_sales' y='item_nbr'))


        # -------------
        # No need maybe
        # -------------

        # '''
        # ### Screen 4 : Expire stock within 10 days
        # '''
        # days with sliders, textarea


        # df_expire = df.loc[df['family_sales'] != 0]\
        #                 .loc[df['family_sales'] <= 10][['family', 'family_sales']]

        # st.dataframe(df_expire)
        # # With selectbox of expire days (10, 9, ...)

        # -------------
        # No need maybe
        # -------------


if st.checkbox('Inv. efficient'):
    '''
    ### Screen down : Inventory Efficient (Lines predict, Lines real)
    '''

    # def display_time_series_2():
    #     # dataframe predicted
    #     fig = px.line(df, x='date', y='family_sales', markers=True)
    #     # dataframe real
    #     fig = px.line(df, x='date', y='family_sales', markers=True)
    #     return fig

    # Just remove after
    st.write(df.groupby(by='family').sum().sort_values('family_sales', ascending=False))


    # df.loc[df['family'] == 'GROCERY I'].loc[df['date'].dt.year == 2015]
    @st.cache
    def inv_efficient_plotly(df):
        # see after to compare predict/real
        # fig = px.line(df, x="date", y="family_sales", color='predict')
        # GROCERY I

        # test between 2 family
        df['date'] = pd.to_datetime(df['date'])
        # df predict (diff with columns predict: True ?)
        df_one_family = df.loc[df['family'] == 'GROCERY I']\
                            .loc[df['date'].dt.year == 2015]
        # df real
        df_second_family = df.loc[df['family'] == 'BEVERAGES']\
                                .loc[df['date'].dt.year == 2015]
        df_compare = pd.concat([df_one_family, df_second_family])

        # later, color to diff predict with real
        fig_one = px.line(df_compare, x='date', y='family_sales', markers=True,
                          title='Inventory Efficient of application', color='family')

        fig_one.update_layout(paper_bgcolor='#B2B1B9')
        # fig_two = px.line(df_second_family, x='date', y='family_sales', markers=True)

        return fig_one
    fig_one = inv_efficient_plotly(df)

    st.plotly_chart(fig_one)



# '''
# ## Map of stores
# '''
# # Map
# # -----------------------
# if mapping:

#     st.write(df['city'].unique().T)

#     '''
#     Screen Map : with Folium (later)
#     '''

#     # st.map()
