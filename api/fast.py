from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import datetime
import pandas as pd
import joblib
from google.cloud import storage
from forecasting_for_sales.params import BUCKET_NAME, BUCKET_STR1_DATA_PATH, BUCKET_STR2_DATA_PATH, STORAGE_LOCATION

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

# defining a root endpoint
@app.get("/")
def index():
    return {"greeting": "Hello world"}


# define a predict endpoint
@app.get("/predict")
def predict(date, store_nbr, family): # laisser tomber date ?
    # charger le csv correspondant au store_nbr pour is_open et is_special ------------------------------ remettre le chemin en ligne quand docké ?
    client = storage.Client()
    #path_str1 = f"gs://{BUCKET_NAME}/{BUCKET_STR1_DATA_PATH}"
    path_str1 = 'holidays_str1.csv'
    holidays_str1 = pd.read_csv(path_str1)
    #path_str2 = f"gs://{BUCKET_NAME}/{BUCKET_STR2_DATA_PATH}"
    path_str2 = 'holidays_str2.csv'
    holidays_str2 = pd.read_csv(path_str2)

    if store_nbr == 1:
        holidays = holidays_str1
    else:
        holidays = holidays_str2

    date = pd.to_datetime(date)

    holidays['date'] = pd.to_datetime(holidays['date'])
    is_open = holidays.loc[holidays['date']==date]['is_open']
    is_special = holidays.loc[holidays['date']==date]['is_special']

    X_dict = {'date': pd.to_datetime(date), # faut-il ajouter une key ???? faut-il laisser tomber la date ?? discussion avec pilou ....
            'is_open': int(is_open),
            'is_special': int(is_special),
    }

    # from dictionary to dataframe
    #df_topredict = pd.DataFrame.from_dict(X_dict)

    # load a model model.joblib trained according to store number and family
    family = family.replace('/', '_')
    model_from_joblib  = joblib.load(f'{STORAGE_LOCATION}/model_{store_nbr}_{family}.joblib')

    # model.predict()
    calc_pred, conf_int = model_from_joblib.predict(start=pd.to_datetime(date), end=(pd.to_datetime(date)+pd.timedelta(days=15)), return_conf_int=True)

    # voir la shape de calc_pred
    # intégrer dans un df avec tout : date, famille, produit (après règle de proportion, inventory units, real sales, predicted sales, nd days inventory outstanding

    # return prediction
    return {"predicted_sales": calc_pred,
            "confidence_int": conf_int
    }
