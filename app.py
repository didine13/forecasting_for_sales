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

# st.dataframe(my_dataframe)
expander_df = st.expander(label='DataFrame')

with expander_df:
    @st.cache
    def get_cached_data():
        # return pd.read_csv('raw_data/train_all_table.csv', nrows=10000).drop(columns='Unnamed: 0')
        # work with 1 store
        return pd.read_csv('raw_data/preprocessed_sales_grouped_21.csv')

    df = get_cached_data()

    option_head = st.slider('head : ', 1, 20000, 5)
    st.write(df.head(option_head))

st.write('Min date: ', min(df['date']))
st.write('Max date: ', max(df['date']))

# Inventory Units (With + or -) %
# Sales Units (With + or -) %
col1, col2 = st.columns(2)
col1.metric("Inventory Units", "437.8", "-$1.25")
col2.metric("Sales Units", "121.10", "0.46%")



with st.sidebar:

    '''
    # Forecast for sales

    ## Management inventory
    '''
    show_plot = st.checkbox('Show Plot')
    '--------'
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

if show_plot:
    # Test directly plot
    # ------------------
    @st.cache
    def countplot_temp():
        # fig , axes = plt.subplots(2, 2, figsize=(15, 16))
        fig , axes = plt.subplots(1, 2, figsize=(15, 16))

        sns.countplot(ax=axes[0], data=df, y='family',
                    order=df['family'].value_counts().index)\
                        .set_title('Countplot family of products')

        # sns.countplot(ax=axes[0, 1], data=df, y='type_x',
        #             order=df['type_x'].value_counts().index)\
        #                 .set_title('Countplot type of products')

        # sns.countplot(ax=axes[1, 0], data=df, y='state',
        #             order=df['state'].value_counts().index)\
        #                 .set_title('Countplot nb products by state')

        # sns.histplot(ax=axes[1],data=df, x='family_sales')\
        #     .set_title('Countplot onpromotion')

        return fig

    st.pyplot(countplot_temp())


# fig2 = px.bar(df[df['store_nbr'] == 25], x='date', y='family_sales')
# st.pyplot()

# ------------------

col_show1, col_show2 = st.columns(2)
# Checkbox to display something
# - Inventory days of supply
# - Product stock details
# - Inventory Trend
# - Inventory Efficient (Lines predict, Lines real)
with col_show1: # Columns 1
    if show_1: # InvDaysSupply,
        '''
        ### Screen 1 : Inventory days of supply
        '''
        # Plot with confidence interval


        '''
        ### Screen 1 : Product stock details
        '''
        # df[['date', 'items', 'family_sales']]
        # [product, date, unit_hand, unit_order]
        # if family_sales actual > or <, print 🔻 or ⬆

        df_stock_details = df[['family', 'family_sales']]\
                            .groupby(by='family').sum()

        st.dataframe(df_stock_details)


    if show_3:
        '''
        ### Screen 3 : Inventory Trend
        '''
        # Lines plots x=date y=family_sales
        def display_time_series():
            # df = px.data.stocks() # replace with your own data source
            fig = px.line(df, x='date', y='family_sales', markers=True)
            return fig

        st.plotly_chart(display_time_series())



with col_show2: # Column 2
    if show_2: # top10, NeedProduct
        '''
        ### Screen 2 : Top 10 of sales, invetories
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
        ### Screen 2 : Needed product (Prod, Alert, Nb)
        '''
        # Select store
        option = st.selectbox('Select a line to filter', df['store_nbr'].unique())
        df_store = df[df['store_nbr'] == option]

        # By store
        st.write(df_store)


    if show_4: # AvailableStock, ExpireStock
        '''
        ### Screen 4 : Available stock by family
        '''
        sb_family = st.selectbox('Family', df['family'].unique())
        df_family = df[df['family'] == sb_family]
        # st.dataframe(df_family[['item_nbr', 'unit_sales']])
        # Better with barplots

        st.plotly_chart(px.bar(df, y='family'))

        # days with sliders, textarea
        '''
        ### Screen 4 : Expire stock within 10 days
        '''


        df_expire = df.loc[df['family_sales'] != 0].loc[df['family_sales'] <= 10]

        st.dataframe(df_expire)
        # With selectbox of expire days (10, 9, ...)


if st.checkbox('Inv. efficient'):
    '''
    ### Screen 3 : Inventory Efficient (Lines predict, Lines real)
    '''

    # def display_time_series_2():
    #     # dataframe predicted
    #     fig = px.line(df, x='date', y='family_sales', markers=True)
    #     # dataframe real
    #     fig = px.line(df, x='date', y='family_sales', markers=True)
    #     return fig

    st.write(df.groupby(by='family').sum().sort_values('family_sales', ascending=False))

    # df.loc[df['family'] == 'GROCERY I'].loc[df['date'].dt.year == 2015]


    @st.cache
    def inv_efficient_plotly(df):
        # see after to compare predict/real
        # fig = px.line(df, x="year", y="lifeExp", color='country')
        # GROCERY I
        df['date'] = pd.to_datetime(df['date'])
        df_one_family = df.loc[df['family'] == 'GROCERY I']\
                            .loc[df['date'].dt.year == 2015]
        df_second_family = df.loc[df['family'] == 'BEVERAGES']\
                                .loc[df['date'].dt.year == 2015]

        fig_one = px.line(df_one_family, x='date', y='family_sales', markers=True,
                          title='Inventory Efficient of application')
        fig_two = px.line(df_second_family, x='date', y='family_sales', markers=True)

        return fig_one, fig_two
    fig_one, fig_two = inv_efficient_plotly(df)

    st.plotly_chart(fig_one)
    st.plotly_chart(fig_two)




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
