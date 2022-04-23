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
from forecasting_for_sales.data import get_train_and_validation_data
import altair as alt


# Initial state of website
st.set_page_config(
    page_title="Forecast for sales",
    page_icon="💰",
    layout="wide")
# css style
CSS = """
.css-fg4pbf, .css-1rh8hwn, h2, h3 {
    color: white;
}

h1 {
    color: orange;
}

.stApp {
    background-color: #2C2E43;
}
.stButton>button {
    height: 6em;
    width: 30em;
    background-color:#A39F9E
}

"""

st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)
# -----------
# Request API
# -----------

# Example request API
# https://favorita-cquq2ssw6q-ew.a.run.app/predict?date=2016-01-01&store_nbr=1&family=BREAD%2FBAKERY

# Call API using `requests`
# Retrieve the prediction from the **JSON** returned by API...
# display the prediction
@st.cache
def get_predict(dict_predict_store):

    # my_url = 'https://docker-tfm-ipbs6r3hdq-ew.a.run.app/predict'
    # url_wagon = 'https://taxifare.lewagon.ai/predict'
    url_forecast = 'https://favorita-cquq2ssw6q-ew.a.run.app/predict'
    response = requests.get(url_forecast, params=dict_predict_store) # my_url
    prediction = response.json()
    return prediction


# Generate DataFrame
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_cached_data():
    # return pd.read_csv('raw_data/train_all_table.csv', nrows=10000).drop(columns='Unnamed: 0')
    # work with 1 store
    # return pd.read_csv('forecasting_for_sales/data/preprocessed_sales_grouped_21.csv')
    return pd.read_csv('forecasting_for_sales/data/preprocessed_sales_grouped_str1.csv')



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

# Plot forecast and confident interval
def plot_forecast(fc, train, test, upper=None, lower=None):
    is_confidence_int = isinstance(upper, np.ndarray) and isinstance(lower, np.ndarray)
    # Prepare plot series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(upper, index=test.index) if is_confidence_int else None
    upper_series = pd.Series(lower, index=test.index) if is_confidence_int else None

    # Plot
    # plt.figure(figsize=(20,10), dpi=100)
    fig, ax = plt.subplots(figsize=(20,12), dpi=100)
    ax.plot(train, label='training', color='black')
    #ax.plot(test, label='actual', color='black', ls='--')
    ax.plot(fc_series, label='forecast', color='orange')
    if is_confidence_int:
        ax.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    ax.set_title('Forecast vs Actuals')
    ax.legend(loc='upper left', fontsize=8);
    return fig


###########################################################################################################################################
##############################################   MAIN  ####################################################################################
###########################################################################################################################################

date_col, family_col, button_col = st.columns(3)
date_unit = date_col.date_input('Choisir la période', datetime.date(2016, 1, 1))
button_ok = button_col.button('Afficher', key="OK")
family = family_col.selectbox(
     'Catégorie de produit',
     ( 'BREAD/BAKERY', 'DAIRY','EGGS', 'MEATS', 'SEAFOOD'))

if button_ok and family == 'BREAD/BAKERY':

    df = get_cached_data()
    df['date'] = pd.to_datetime(df['date'])

    sb_month_unit = date_unit.month
    sb_year_unit = date_unit.year

    seuil_d_alerte = 15

    f'''
    # Prévisions des ventes pour l'année {sb_year_unit}
    ---
    '''

    su_actual_month, su_past_month = inventory_unit(sb_month_unit, sb_year_unit)
    su_past_month = -(100 - (su_actual_month * 100 / su_past_month))

    col1, col2 = st.columns(2)



    # Call API
    dict_predict_store = {
        'date': date_unit,
        'store_nbr': 1,
        'family': family
    }
    prediction = get_predict(dict_predict_store)
    prediction = {'predicted_sales-per_item': '{"columns":["item_nbr","forecast_product"],"index":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],"data":[[103665,-0.1466762013],[153239,0.1123254223],[153395,0.5495556805],[153398,1.448063851],[165718,0.6940110239],[215370,0.93472928],[253103,0.0],[265279,1.5096506299],[269084,0.6834305765],[302952,0.7299163967],[310644,5.6559958399],[310647,0.5820810254],[311994,9.4483281797],[312113,0.7590647999],[315473,0.0],[315474,1.3839657389],[359913,1.28872513],[360313,0.9708956844],[360314,2.2992251188],[402299,0.0]]}',
                'confidence_int': '[[369.6538898542785, 528.230879093483], [323.378085574303, 481.9550748135076], [449.69862452750857, 608.2756137667131], [300.01599841998194, 458.5929876591866], [332.44847349191446, 491.0254627311191], [301.228373190773, 459.80536242997766], [316.5217120301507, 475.09870126935533], [335.96638802943323, 494.54337726863787], [298.78129636328543, 457.35828560249007], [296.1490519126586, 454.7260408574747], [295.1983801188044, 453.7753690636205], [304.89968506573246, 463.47667401054855], [360.3411007282706, 521.2829847195223], [241.48354586765439, 402.4254298589062], [371.2996912904979, 532.2415752817496], [342.32048914188044, 503.2623731331323], [272.78879209171123, 433.7306760829631], [331.0450673370848, 491.98695132833666], [262.06854740783285, 423.0104313990847], [315.57650296302154, 476.5183869542734]]',
                'family_forecast': '[448.9423844738808, 402.6665801939053, 528.9871191471109, 379.30449303958426, 411.7369681115168, 380.51686781037535, 395.810206649753, 415.25488264903555, 378.06979098288775, 375.43754638506664, 374.4868745912124, 384.1881795381405, 440.81204272389647, 321.9544878632803, 451.77063328612377, 422.7914311375064, 353.25973408733716, 411.5160093327107, 342.5394894034588, 396.0474449586475]'}


    df_proportion = pd.read_csv('forecasting_for_sales/data/proportions_str1.csv') #, usecols=['item_sales'])
    df_proportion = df_proportion[(df_proportion['family'] == family) & (df_proportion['month'] == 1)].reset_index().head(20)
    df_to_plot_3 = pd.read_csv('forecasting_for_sales/data/toplot3.csv')
    df_to_plot_3['date'] = pd.to_datetime(df_to_plot_3['date'])

    # predicted_sales_per_item
    predicted_sales_per_item = json.loads(prediction['predicted_sales-per_item'])
    #predicted_sales_per_item

    confidence_int = np.array(json.loads(prediction['confidence_int'])) # confidence_int
    # confidence_int
    family_forecast = np.array(json.loads(prediction['family_forecast'])) # forecast
    #family_forecast

    train_test = get_train_and_validation_data(dict_predict_store['store_nbr'], plot_fc=True, family=family, nb_train=90, nb_val=len(family_forecast))
    train=train_test['train']
    test = train_test['test']


    col1.write('## Gestion des ventes')
    col2.write('## Gestion d\'inventaires')

    # Sales Units (With + or -) %
    try:
        col1.metric("Unités de ventes", f"{round(su_actual_month)}", f"{round(su_past_month)}%")
    except:
        col1.metric("Unités de ventes", f"{su_actual_month}", f"{su_past_month}%")


    col_left, col_right = st.columns(2)
    with col_left:
        '### Prévisions des ventes (Boulangerie/Pâtisserie)'
        # Plot with confidence interval - start

        st.pyplot(plot_forecast(family_forecast, train, test, confidence_int[:,0], confidence_int[:,1]))


        '### Top 10 des ventes (Boulangerie/Pâtisserie)'

        # predicted_sales_per_item['data'] # item_nbr, forecast_product
        df_pred_sales_per_item = pd.DataFrame(predicted_sales_per_item['data'],
                                            columns =['item_nbr', 'forecast_product'])

        #st.dataframe(df_pred_sales_per_item.sort_values(by='forecast_product', ascending=False))
        df_pred_sales_per_item.sort_values(by='forecast_product', ascending=False, inplace=True)

        df_pred_sales_per_item['item_nbr'] = df_pred_sales_per_item['item_nbr'].astype(str)


        df_pred_sales_per_item = df_pred_sales_per_item.groupby(by='item_nbr').sum()\
                                .sort_values(by='forecast_product').tail(10)
        fig = px.bar(df_pred_sales_per_item,
                    x='forecast_product',
                    y=df_pred_sales_per_item.index,
                    orientation='h',
                    labels={'forecast_product': 'Nombres des ventes','item_nbr': 'Produits'})

        st.write(fig)



    with col_right:
        '### Inventaire restant en jours'

        # Stock cumulé restant en nombre de jours
        st.bar_chart(df_to_plot_3[['Inventory']].rename(columns={'Inventory':'Stock'}), height=600)

        # For forecast   bar Stact prédiction/stock

        # forecast_product_20 = np.array([row[1] for row in predicted_sales_per_item['data']]).reshape(20, 1)
        # df_proportion_last_20 = df_proportion[['item_sales']].tail(20).values
        # df_stock = pd.DataFrame(np.hstack((forecast_product_20, df_proportion_last_20)), columns=['initial_stock', 'pred'])
        # df_stock['current_stock'] = df_stock['initial_stock'] - df_stock['pred']
        # df_stock['stock_pred'] = df_stock['initial_stock'] + df_stock['pred']
        # st.bar_chart(df_stock[['pred', 'current_stock']], height=600)


        '### Stock à renouveller'

        # La colone Inventory est cumulée (inventaire au 1er jour), donc décroit ds le temps. On par d'une max qui décroit
        # On prendra la totalité de forecast qui sera ~ forecast cumulé (au 1er jour)
        df_to_plot_3['Inventory'] = df_to_plot_3['Inventory'].fillna(0)
        df_to_plot_3.drop(columns=['BREAD/BAKERY forecast','cumulated forecast', 'BREAD/BAKERY history'], inplace=True)
        df_to_plot_3['month']= df_to_plot_3['date'].dt.month
        df_to_plot_3['year']= df_to_plot_3['date'].dt.year
        df_to_plot_3 = df_to_plot_3[(df_to_plot_3['month'] == 1) & (df_to_plot_3['year']==2016)].reset_index()
        df_proportion['inventory_product'] = df_proportion['proportion_item_to_family'] * df_to_plot_3.loc[0,'Inventory']
        df_proportion['forecast_product'] = df_proportion['proportion_item_to_family'] * family_forecast.sum()
        #df_proportion['diff'] = (df_proportion['inventory_product'] - df_proportion['forecast_product']).round(decimals=2)
        df_proportion['diff'] = (df_proportion['inventory_product'] - df_proportion['forecast_product']).round()

        df_expired_product_stock = df_proportion[df_proportion['diff'] <= seuil_d_alerte][['item_nbr','item_sales', 'proportion_item_to_family', 'inventory_product', 'forecast_product', 'diff']] \
                                    .sort_values(by='diff') \
                                    .rename(columns={"item_nbr": "Produits", "diff": "Stock restant"})
        df_expired_product_stock['Produits'] = df_expired_product_stock['Produits'].astype(str)

        base = alt.Chart(df_expired_product_stock).encode(
            y='Produits',
            x='Stock restant',
        )

        bars = base.mark_bar().encode(
            color='Produits'
        )

        text = base.mark_text(
            align='left',
            baseline='middle',
            dx=3,
            #color='black'
        ).encode(
            text='Stock restant'
        )
        #-------------------   ou   -------------------------------------
        # bars = alt.Chart(df_expired_product_stock).mark_bar().encode(
        #     y='item_nbr',
        #     x='diff',
        #     color='item_nbr'
        # )
        # text = bars.mark_text(
        #     align='left',
        #     baseline='middle',
        #     dx=3,
        #     color='black'
        # ).encode(
        #     text='diff'
        # )
        st.altair_chart(bars + text, use_container_width=True)



        command = col_right.button('COMMANDER', key="command", disabled=True)
        if command:
            col_right.markdown('<p style="font-size:40px; display: none">Commande effectuée</p>', unsafe_allow_html=True)


elif family != "BREAD/BAKERY":
    st.write('Pas de donnée suffisante pour pouvoir vous conseiller')
else:
    st.write('Selectionner une date et une famille de produits')
