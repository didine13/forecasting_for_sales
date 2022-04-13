# Import
# from matplotlib.font_manager import get_fontconfig_fonts
import streamlit as st
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
    color: orange;
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
    df['date'] = pd.to_datetime(df['date'])
    # df = df[df['date'].dt.year == 2017]


    option_head = st.slider('head : ', 1, 1000, 5)
    st.write(df.head(option_head))

st.write('Min date: ', min(df['date']))
st.write('Max date: ', max(df['date']))


# check actual month (vs previous month)
# compare sum 2 family_sales
# show with deficit or benefit

sb_month_unit = st.selectbox('Month Unit', range(min(df['date'].dt.month),
                                                max(df['date'].dt.month)))

sb_year_unit = st.selectbox('Year Unit', range(min(df['date'].dt.year),
                                                max(df['date'].dt.year)))

def inventory_unit(sb_month_unit, sb_year_unit):
    df_present = df.loc[(df['date'].dt.year == sb_year_unit) & (df['date'].dt.month == sb_month_unit)]
    result_actual = df_present['family_sales'].sum()

    if sb_month_unit == 1:
        df_past = df.loc[(df['date'].dt.year == sb_year_unit - 1) &
                        (df['date'].dt.month == 12)]
    else:
        df_past = df.loc[(df['date'].dt.year == sb_year_unit) &
                        (df['date'].dt.month == sb_month_unit - 1)]
    result_past = result_actual - df_past['family_sales'].sum()


    return result_actual, result_past


# Inventory Units (With + or -) %
# Sales Units (With + or -) %
col1, col2 = st.columns(2)
ui_actual_month, iu_past_month = inventory_unit(sb_month_unit, sb_year_unit)
col1.metric("Sales Units", f"{ui_actual_month}", f"{iu_past_month}")

# In Progress
# Need stock
col2.metric("Inventory Units", "121.10", "0.46%")


with st.sidebar:
    '''
    # Forecast for sales

    ## Management inventory
    ---
    '''


    show_1 = st.checkbox('Show 1')
    '''
    - Needed product (Prod, Alert, Nb)
    ---
    '''

    show_2 = st.checkbox('Show 2')
    '''
    - Top 10 of sales, inventories
    ---
    '''

    show_3 = st.checkbox('Show 3')
    '''
    - Inventory Trend
    ---
    '''

    show_4 = st.checkbox('Show 4')
    '''
    - Available stock by Family
    - Forecast Sales
    ---
    '''

    show_inv_eff = st.checkbox('Inv. efficient')
    '''
    - Inventory Efficient (Lines predict/Real)
    '''

    # '''
    # Mapping
    # '''
    # mapping = st.checkbox('Show Map')


col_left, col_right = st.columns(2)
with col_left: # Columns left
    if show_1: #  Needed product


        '### Screen 1 left : Needed product (Prod, Alert, Nb)'

        # color not functionnal
        # def _color_red_or_green(val):
        #     color = 'red' if val < 10 else 'green'
        #     return 'color: %s' % color

        # Select store - see after
        # option = st.selectbox('Select a line to filter', df['store_nbr'].unique())
        # df_store = df[df['store_nbr'] == option]
        df_store = df

        df_store = df_store[['family', 'family_sales']]\
                            .groupby(by='family').sum()\
                            .sort_values('family_sales', ascending=False)
        df_store['alert'] = (df_store['family_sales'] <= 0)

        # add colors
        # df_store.style.apply(_color_red_or_green,
        #                      subset='family_sales', axis=1)
        # By store
        st.dataframe(df_store[['alert', 'family_sales']])


    if show_3: # InvTrend
        '### Screen 3 left : Inventory Trend'
        # Selectbox for year and family
        sb_year_inv = st.selectbox('Year inv',
                        range(min(df['date'].dt.year),
                            max(df['date'].dt.year)))
        sb_family_inv = st.selectbox('Family inv', df['family'].unique())

        @st.cache(suppress_st_warning=True, allow_output_mutation=True)
        def display_time_series(df, sb_family_inv, year):
            # df = px.data.stocks() # replace with your own data source

            # x=date y=family_sales by year by family
            df_family = df.loc[df['family'] == sb_family_inv]\
                            .loc[df['date'].dt.year == year]
            # Lines plots
            fig = px.line(df_family, x='date', y='family_sales', markers=True)
            fig.update_layout(paper_bgcolor='#B2B1B9')
            return fig


        st.plotly_chart(display_time_series(df, sb_family_inv, sb_year_inv))



with col_right: # Column right
    if show_2: # top10, AvailableStock, Forecast Sales
        '''
        ### Screen 2 right : Top 10 of sales, inventories
        '''
        # add date to choose
        sb_year_top_10 = st.selectbox('Year',
                                range(min(df['date'].dt.year),
                                    max(df['date'].dt.year)))

        # Later to select meat, chicken, beef, etc.
        # st.multiselect(label="", options=avg_wine_df.columns.tolist(), default=["alcohol","malic_acid"])

        def show_top_10(df, sb_year_top_10):
            df_in_date = df[df['date'].dt.year == sb_year_top_10]
            df_top_10 = df_in_date[['family', 'family_sales']].groupby(by='family')\
                                .sum()\
                                .sort_values('family_sales', ascending=False).head(10)
            return df_top_10


        st.dataframe(show_top_10(df, sb_year_top_10))




    if show_4: # AvailableStock, Forecast Sales
        '### Screen right : Available stock by family'

        '''
        Add someting here
        Need item_nbr if check 1 family...
        '''
        # sb_family_stock = st.selectbox('Family', df['family'].unique())

        # df_family = df[df['family'] == sb_family_stock]
        # df_family = df[df['family'] == sb_family_stock]
        # st.plotly_chart(px.bar(df_family,x='unit_sales' y='item_nbr'))



        '### Screen 4 right :  Forecast Sales'
        # Plot with confidence interval - start

        'Something here, a Plot with confidence interval'

        # -------------
        # No need maybe
        # -------------

        # # Create a correct Training/Test split to predict the last 50 points
        # train = df['linearized'][0:150]
        # test = df['linearized'][150:]

        # # Build Model
        # arima = ARIMA(train, order=(0, 1, 1))
        # arima = arima.fit()

        # # Forecast
        # forecast, std_err, confidence_int = arima.forecast(len(test), alpha=0.05)  # 95% confidence

        # -------------
        # No need maybe
        # -------------

        def plot_forecast(fc, train, test, upper=None, lower=None):
            is_confidence_int = isinstance(upper, np.ndarray) and isinstance(lower, np.ndarray)
            # Prepare plot series
            fc_series = pd.Series(fc, index=test.index)
            lower_series = pd.Series(upper, index=test.index) if is_confidence_int else None
            upper_series = pd.Series(lower, index=test.index) if is_confidence_int else None

            # Plot
            plt.figure(figsize=(10,4), dpi=100)
            plt.plot(train, label='training', color='black')
            plt.plot(test, label='actual', color='black', ls='--')
            plt.plot(fc_series, label='forecast', color='orange')
            if is_confidence_int:
                plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
            plt.title('Forecast vs Actuals')
            plt.legend(loc='upper left', fontsize=8);

        # Plot with confidence interval
        # plot_forecast(forecast, train, test, confidence_int[:,0], confidence_int[:,1])

        # Plot with confidence interval - end



if show_inv_eff: # InvEff (Pred, Real)
    '### Screen down : Inventory Efficient (Lines predict, Lines real)'

    # def display_time_series_2():
    #     # dataframe predicted
    #     fig = px.line(df, x='date', y='family_sales', markers=True)
    #     # dataframe real
    #     fig = px.line(df, x='date', y='family_sales', markers=True)
    #     return fig

    # Just remove after
    st.write(df.groupby(by='family').sum().sort_values('family_sales', ascending=False))


    # df.loc[df['family'] == 'GROCERY I'].loc[df['date'].dt.year == 2015]
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
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

        # later, color= to diff predict with real
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
