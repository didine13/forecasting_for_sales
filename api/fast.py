import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import pandas as pd
import joblib
#from google.cloud import storage
from forecasting_for_sales.params import BUCKET_NAME
from forecasting_for_sales.data_gcp import *
from os.path import exists



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def generate_inventory(calc_pred):
    pass

@app.get("/")
def index():
    return {"greeting": "Hello world"}


# define a root `/` endpoint
@app.get("/predict")
def predict(date, store_nbr, family):
    #holidays_str1 = get_data_from_gcp("preprocessed_sales_grouped_str1.csv")
    #holidays_str2 = get_data_from_gcp("preprocessed_sales_grouped_str2.csv")
    holidays_str1 = pd.read_csv(f"forecasting_for_sales/data/preprocessed_sales_grouped_str1.csv")
    proportion = pd.read_csv(f"forecasting_for_sales/data/proportions_str{store_nbr}.csv")

    if int(store_nbr) == 1:
        holidays = holidays_str1
    else:
        #holidays = holidays_str2
        return {"error":"No holidays file"}

    print(holidays_str1.dtypes)

    holidays = holidays[['date','is_open', 'is_special']].groupby('date').sum().reset_index()
    is_open = holidays.loc[holidays['date']==date]['is_open'].tolist()[0]
    is_special = holidays.loc[holidays['date']==date]['is_special'].tolist()[0]

    if is_open >= 2 :
        is_open = 1
    if is_special >= 2:
        is_special = 1

    X_dict = pd.DataFrame({
            'is_open': [int(is_open)],
            'is_special': [int(is_special)],
            'x0_DAIRY' : [0],
            'x0_DELI': [0],
            'x0_EGGS': [0],
            'x0_MEATS': [0],
            'x0_POULTRY': [0],
            'x0_PREPARED FOODS': [0],
            'x0_PRODUCE': [0],
            'x0_SEAFOOD':[0],
            'dcoilwtico':[0]
    })

    # load a model model.joblib trained according to store number and family
    model_name = f"model_{store_nbr}_x0_{family}.joblib"
    model_name = model_name.replace('/','_')


    if exists(model_name):
        print(colored(f"file {model_name} file exists locally ...",
                        "green"))
        model_from_joblib  = joblib.load(model_name)
    else :
        model_from_joblib = download_model_from_gcp(model_name)
        print(colored(f"file {model_name} downloaded from GCP OK ...",
                        "green"))

    if model_from_joblib == False:
        return {"error":f"Pas de modèle entrainé pour la catégorie {family}"}

    forecast, conf_int = model_from_joblib.predict(n_periods=20,
                                                   X=X_dict.values,
                                                   return_conf_int=True)

    family_items = proportion[(proportion['family'] == family) &
                              (proportion['month'] == int(pd.Series(pd.to_datetime(date)).dt.month))].drop(columns="Unnamed: 0").reset_index()

    family_forecast = pd.DataFrame(forecast)
    family_forecast.head(len(family_forecast))
    family_items = proportion[(proportion['family'] == family) &
                              (proportion['month'] == int(pd.Series(pd.to_datetime(date)).dt.month))].drop(columns="Unnamed: 0").reset_index() \
                          .head(len(family_forecast))

    family_items['forecast_product'] = family_items['proportion_item_to_family'] * family_forecast[0]

    return {"predicted_sales-per_item": family_items[['item_nbr', 'forecast_product']].to_json(orient='split'),
            "confidence_int": json.dumps(conf_int.tolist()),
            "family_forecast" : json.dumps(forecast.tolist())
    }

# date="2016-01-01"
# family="BREAD/BAKERY"
# store=1
# print(predict(date, store, family))
