MODEL_NAME = 'forecasting_for_sales'
MODEL_VERSION = 'v1'
#STORAGE_LOCATION = 'models/forecasting_for_sales/model.joblib'
BUCKET_NAME = 'wagon-data-835-forecasting-for-sales'
BUCKET_TRAIN_DATA_PATH = 'data'
#BUCKET_TRAIN_DATA_PATH = 'data/train.csv'
STORAGE_LOCATION = 'models/forecasting_for_sales'
BUCKET_STR1_DATA_PATH = 'data'
BUCKET_STR2_DATA_PATH = 'data/preprocessed_sales_grouped_str2.csv'


### MLFLOW configuration - - - - - - - - - - - - - - - - - - -

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[country] [city] [user] model + version"
