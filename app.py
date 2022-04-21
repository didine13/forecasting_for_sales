# Import
import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime

import requests
import json
# from PIL import Image


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
.stButton>button {
    height: 6em;
    width: 30em;
}
"""

st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)
# -----------
# Request API
# -----------

# pickup_datetime = f"{date} {time}"

date_predict = '2016-01-01'
store_nbr_predict = 1
family_predict = 'BREAD/BAKERY' # BREAD%2FBAKERY
# family_predict = '' # BREAD%2FBAKERY

dict_predict_store = {
    'date': date_predict,
    'store_nbr': store_nbr_predict,
    'family': family_predict
}

# Example request API
# https://favorita-cquq2ssw6q-ew.a.run.app/predict?date=2016-01-01&store_nbr=1&family=BREAD%2FBAKERY

# Call API using `requests`
# Retrieve the prediction from the **JSON** returned by API...
# display the prediction
@st.cache
def get_predict():

    # my_url = 'https://docker-tfm-ipbs6r3hdq-ew.a.run.app/predict'
    # url_wagon = 'https://taxifare.lewagon.ai/predict'
    url_forecast = 'https://favorita-cquq2ssw6q-ew.a.run.app/predict'
    response = requests.get(url_forecast, params=dict_predict_store) # my_url
    sales_fare = response.json()
    return sales_fare

sales_fare = get_predict()
# -----------
# Request API
# -----------

# sales_fare['predicted_sales-per_item']['data'] # top 10

# sales_fare
predicted_sales_per_item = json.loads(sales_fare['predicted_sales-per_item'])
# predicted_sales_per_item

confidence_int = json.loads(sales_fare['confidence_int']) # confidence_int
# confidence_int
family_forecast = json.loads(sales_fare['family_forecast']) # forecast
# family_forecast


# Generate DataFrame
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_cached_data():
    # return pd.read_csv('raw_data/train_all_table.csv', nrows=10000).drop(columns='Unnamed: 0')
    # work with 1 store
    # return pd.read_csv('forecasting_for_sales/data/preprocessed_sales_grouped_21.csv')
    return pd.read_csv('forecasting_for_sales/data/preprocessed_sales_grouped_1.csv')

df = get_cached_data()
df['date'] = pd.to_datetime(df['date'])

'''
# Prévisions des ventes pour l'année 2016
---
'''



# check actual month (vs previous month)
# compare sum 2 family_sales
# show with deficit or benefit


def inventory_unit(sb_month_unit, sb_year_unit):
    df_present = df.loc[(df['date'].dt.year == sb_year_unit) & (df['date'].dt.month == sb_month_unit)]
    result_actual = df_present['family_sales'].sum()

    if sb_month_unit == 1:
        df_past = df.loc[(df['date'].dt.year == sb_year_unit - 1) &
                        (df['date'].dt.month == 12)]
    else:
        df_past = df.loc[(df['date'].dt.year == sb_year_unit) &
                        (df['date'].dt.month == sb_month_unit - 1)]
    # result_past = result_actual - df_past['family_sales'].sum()
    result_past = df_past['family_sales'].sum()


    return result_actual, result_past


col1, col2 = st.columns(2)

col1.write('## Gestion des ventes')
col2.write('## Gestion d\'inventaires')


date_unit = col1.date_input('Choisir la période', datetime.date(2015, 8, 1))

sb_month_unit = date_unit.month
sb_year_unit = date_unit.year

su_actual_month, su_past_month = inventory_unit(sb_month_unit, sb_year_unit)
su_past_month = -(100 - (su_actual_month * 100 / su_past_month))

# Sales Units (With + or -) %
try:
    col1.metric("Unités de ventes", f"{round(su_actual_month)}", f"{round(su_past_month)}%")
except:
    col1.metric("Unités de ventes", f"{su_actual_month}", f"{su_past_month}%")


col_left, col_right = st.columns(2)
with col_left:
    '### Prévisions des ventes (Boulangerie/Pâtisserie)'
    # Plot with confidence interval - start
    # forecast, std_err, confidence_int = {'predicted_sales': '{"columns":["item_nbr","forecast_product"],"index":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],"data":[[103665,118.1805681175],[153239,12.0317315362],[153395,45.4006592073],[153398,119.9685781917],[165718,63.6514990944],[215370,111.586895726],[253103,0.0],[265279,71.2477012761],[269084,59.2145852272],[302952,67.2229645917],[310644,634.6395670119],[310647,46.7342347702],[311994,874.0568623078],[312113,71.9110566686],[315473,0.0],[315474,111.4320607333],[359913,124.2214658545],[360313,98.6543577846],[360314,217.5643155294],[402299,0.0]]}', 'confidence_int': {'6602.960957236907': 15542.673115061367, '5626.846602575789': 14566.55876040025, '6348.378908161257': 15288.091065985718, '6250.10062934992': 15189.812787174382, '5641.551794978621': 14581.26395280308, '7109.3697105270085': 16049.08186835147, '6463.915887184812': 15403.62804500927, '1248.0240571079748': 10187.736214932436, '4936.72242802496': 13876.43458584942, '5176.752296871691': 14116.46445469607, '6850.997021333551': 15790.70917915793, '5552.547742131246': 14492.259899955625, '5399.764680278498': 15199.011466229273, '5381.514230917039': 15180.761016867813, '5753.044452900291': 15552.291238851065, '5366.474603953162': 15165.721389903936, '4633.559635709262': 14432.806421660036, '5797.710271910317': 15596.957057861091, '5840.027843573505': 15639.27462952428, '3276.965667413988': 13076.212453364762}, 'family_predictions': '{"columns":[0],"index":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],"data":[[11072.8170361491],[10096.702681488],[10818.2349870735],[10719.9567082622],[10111.4078738909],[11579.2257894392],[10933.771966097],[5717.8801360202],[9406.5785069372],[9646.6083757839],[11320.8531002457],[10022.4038210434],[10299.3880732539],[10281.1376238924],[10652.6678458757],[10266.0979969285],[9533.1830286846],[10697.3336648857],[10739.6512365489],[8176.5890603894]]}'}

    train_for_plot_forecast = pd.read_csv('forecasting_for_sales/data/train_14042022.csv')
    var_forecast = pd.read_csv('forecasting_for_sales/data/forecast.csv')

    to_plot = pd.read_csv('forecasting_for_sales/data/toplot.csv')
    to_plot['date'] = pd.to_datetime(to_plot['date'])
    to_plot = to_plot.loc[(to_plot['date'] >= '2015-10-01') & (to_plot['date'] <= '2016-01-31')]

    # 1 octobre et 31 janv

    fig_plot, ax = plt.subplots()
    ax.plot(to_plot['date'], to_plot['BREAD/BAKERY forecast'])
    ax.plot(to_plot['date'], to_plot['BREAD/BAKERY history'])

    ax.set_facecolor('#2C2E43')
    ax.xaxis.label.set_color('#ffffff')
    fig_plot.set_facecolor('#2C2E43')

    st.pyplot(fig_plot)


    # plt.plot(to_plot)

    # st.line_chart(to_plot)

    var_forecast.rename(columns={'Unnamed: 0': 'date', '0':'x0_BREAD/BAKERY' }, inplace=True)

    train_for_plot_forecast['date'] = pd.to_datetime(train_for_plot_forecast['date'])
    var_forecast['date'] = pd.to_datetime(var_forecast['date'])



    # st.line_chart(train_for_plot_forecast, var_forecast)


    # Code copied in lesson
    def plot_forecast(fc, train, test, upper=None, lower=None):
        is_confidence_int = isinstance(upper, np.ndarray) and isinstance(lower, np.ndarray)
        # Prepare plot series
        fc_series = pd.Series(fc, index=test.index)
        lower_series = pd.Series(upper, index=test.index) if is_confidence_int else None
        upper_series = pd.Series(lower, index=test.index) if is_confidence_int else None

        # Plot
        plt.figure(figsize=(10,4), dpi=100)
        plt.plot(train, label='training', color='black')
        plt.plot(test, label='reel', color='black', ls='--')
        plt.plot(fc_series, label='forecast', color='orange')
        if is_confidence_int:
            plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
        plt.title('Prédictions vs Réel')
        plt.legend(loc='upper left', fontsize=8);

    # # create df_store_1 for test function predict for BREAD/BAKERY
    # df_store_1 = pd.read_csv('forecasting_for_sales/data/preprocessed_sales_grouped_1.csv')
    # df_store_1 = df_store_1[df_store_1['family'] == 'BREAD/BAKERY']
    # df_store_1['date'] = pd.to_datetime(df_store_1['date'])

    # # test df about BREAD/BAKERY

    # # Prepare train and test
    # mask_train = (df_store_1['date'] >= '2015-09-01') & (df_store_1['date'] <= '2015-12-31')

    # mask_test = (df_store_1['date'] >= '2016-01-01') & (df_store_1['date'] <= '2016-01-20')


    # train_store_1 = df_store_1.loc[df_store_1['family'] == 'BREAD/BAKERY']\
    #                             .loc[mask_train] # 2013 -> 2015


    # test_store_1 = df_store_1.loc[df_store_1['family'] == 'BREAD/BAKERY']\
    #                             .loc[mask_test] # 2016 ->

    # Plot with confidence interval
    # plot_forecast(forecast, train_store_1, test_store_1, confidence_int[:,0], confidence_int[:,1])
    # var_plot_forecast = plot_forecast(family_forecast,
    #                 train_for_plot_forecast['family_sales'], # x0_BREAD_BAKERY
    #                 test_for['family_sales'],
    #                 [row[0] for row in confidence_int],
    #                 [row[1] for row in confidence_int])


    # st.pyplot(var_plot_forecast)

    # image
    # image = Image.open('forecasting_for_sales/data/image2.png')

    # st.image(image, caption='Prévisions des ventes - Graphe')



    '### Top 10 des ventes (Boulangerie/Pâtisserie)'

    # predicted_sales_per_item['data'] # item_nbr, forecast_product
    df_pred_sales_per_item = pd.DataFrame(predicted_sales_per_item['data'],
                                          columns =['item_nbr', 'forecast_product'])

    # st.dataframe(df_pred_sales_per_item)

    df_pred_sales_per_item['item_nbr'] = df_pred_sales_per_item['item_nbr'].astype(str)

    def barplot_top10(df):
        # fig = px.bar(df, x='family_sales', y=df.index, color='item_nbr')
        df = df.groupby(by='item_nbr').sum()\
                            .sort_values('item_nbr', ascending=False).head(10)
        fig = px.bar(df, x='forecast_product', y=df.index, labels={'forecast_product': 'Nombres des ventes',
                                                                   'item_nbr': ''})

        fig.update_layout(paper_bgcolor='#2C2E43', font_size=24, plot_bgcolor='#2C2E43')
        return fig

    st.plotly_chart(barplot_top10(df_pred_sales_per_item))







with col_right:
    '### Inventaire restant en jours'



    # For forecast
    df_gr_1 = pd.read_csv('forecasting_for_sales/data/preprocessed_sales_grouped_1.csv')
    df_proportion = pd.read_csv('forecasting_for_sales/data/proportion_bread.csv', usecols=['item_sales'])

    df_to_plot_3 = pd.read_csv('forecasting_for_sales/data/toplot3.csv')
    df_to_plot_3['date'] = pd.to_datetime(df_to_plot_3['date'])


    # predicted_sales_per_item['data']
    df_gr_1 = df_gr_1[df_gr_1['family'] == 'BREAD/BAKERY']


    forecast_product_20 = np.array([row[1] for row in predicted_sales_per_item['data']]).reshape(20, 1)
    df_proportion_last_20 = df_proportion.tail(20).values


    df_stock = pd.DataFrame(np.hstack((forecast_product_20, df_proportion_last_20)), columns=['initial_stock', 'pred'])

    # np.array(forecast_product_20).reshape(len(forecast_product_20), 1)
    # df_stock = pd.DataFrame(
    #     {'initial_stock': df_proportion_last_20,
    #      'pred': forecast_product_20
    #      }, index=[range(20)])


    df_stock['current_stock'] = df_stock['initial_stock'] - df_stock['pred']

    df_stock['stock_pred'] = df_stock['initial_stock'] + df_stock['pred']

    # def barplot_matplot_bread(df):
    #     fig, ax = plt.subplots()

    #     ax.plot(df, kind='bar', stacked=True)
    #     return fig

    # fig, ax = plt.subplots()

    # st.pyplot(barplot_matplot_bread(df_stock.loc[:,['pred', 'current_stock']]))

    # df_stock.loc[:,['pred', 'current_stock']].plot(kind='bar', stacked=True)

    # st.pyplot(fig)


    # st.dataframe(df_stock)
    # st.dataframe(df_stock.loc[:,['pred', 'current_stock']])


    # st.bar_chart(df_stock.loc[:,['current_stock']], height=600)
    # df_to_plot_3 = df_to_plot_3[df_to_plot_3['date'].dt.year == 2016]
    #st.pyplot(df_to_plot_3.loc['2016-01-01':'2016-01-31','Inventory'])

    chart_data2 = df_to_plot_3['Inventory']
    st.bar_chart(chart_data2, height=600)

    # plt.plot(df_stock.loc[:,['pred', 'current_stock']], kind='bar', stack)

    # ------------------------------


    # value : number of days left (with stock actual )

    # bread_barkery_dict = {
    #     'Pain de mie': [8, 'Bon'],
    #     'Pain': [2, 'Alerte'],
    #     'Croissant': [10, 'Bon'],
    #     'Muffin': [8, 'Bon'],
    #     'Tarte aux pommes': [5, 'Alerte'],
    #     'Tarte aux chocolats': [6, 'Alerte'],
    #     'Sablé aux chocolats': [7, 'Bon'],
    #     'Bretzel': [3, 'Alerte']
    #     }

    # df_bread_barkery = pd.DataFrame.from_dict(bread_barkery_dict, orient='index', columns=['nb_days_left', 'alert'])


    # st.dataframe(df_bread_barkery)
    def barplot_bread(df_bread):
        fig = px.bar(df_bread, x='current_stock', y=df_bread.index, color='alert', labels={'nb_days_left':'Nombres de jours restants',
                                                                           'index': ''})

        fig.update_layout(paper_bgcolor='#2C2E43',
                          font_size=24,
                          plot_bgcolor='#2C2E43',
                        #   legend_bgcolor='#2C2E43',
                          showlegend=False)
        return fig

    # Plot Inventory
    # st.plotly_chart(barplot_bread(df_bread_barkery))

    if st.button('COMMANDER'):
        st.markdown('<p style="font-size:40px;">Commande effectuée</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:40px;">Commande non effectuée</p>', unsafe_allow_html=True)
