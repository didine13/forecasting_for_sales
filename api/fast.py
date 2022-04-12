from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import joblib
from google.cloud import storage
from forecasting_for_sales.params import BUCKET_NAME, BUCKET_STR1_DATA_PATH, BUCKET_STR2_DATA_PATH

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def generate_inventory(calc_pred):


# define a root `/` endpoint
@app.get("/predict")
def predict(date, store_nbr, family):

    # charger le csv correspondant au store_nbr pour is_open et is_special
    path_str1 = f"gs://{BUCKET_NAME}/{BUCKET_STR1_DATA_PATH}"
    holidays_str1 = pd.read_csv(path_str1)
    path_str2 = f"gs://{BUCKET_NAME}/{BUCKET_STR2_DATA_PATH}"
    holidays_str2 = pd.read_csv(path_str2)

    if store_nbr == 1:
        holidays = holidays_str1
    else:
        holidays = holidays_str2

    is_open = holidays.loc[holidays['date']==date]['is_open']
    is_special = holidays.loc[holidays['date']==date]['is_special']

    X_dict = {'date': pd.to_datetime(date),
            'store_nbr': int(store_nbr),
            'family': family,
            'is_open': int(is_open),
            'is_special': int(is_special),
    }

    # from dictionary to dataframe
    df_topredict = pd.DataFrame.from_dict(X_dict)

    # load a model model.joblib trained according to store number and family
    family = family.replace('/', '_')
    model_from_joblib  = joblib.load(f'model_{store_nbr}_{family}.joblib')

    # model.predict()
    calc_pred = model_from_joblib.predict(df_topredict)[0]

    # return prediction
    return {"predicted_sales": calc_pred}
