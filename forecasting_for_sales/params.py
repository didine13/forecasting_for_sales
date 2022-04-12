MODEL_NAME = 'forecasting_for_sales'
MODEL_VERSION = 'v1'
#STORAGE_LOCATION = 'models/forecasting_for_sales/model.joblib'
BUCKET_NAME = 'wagon-data-835-forecasting-for-sales'
BUCKET_TRAIN_DATA_PATH = 'data/train2016.csv'

### MLFLOW configuration - - - - - - - - - - - - - - - - - - -

MLFLOW_URI = "https://mlflow.lewagon.ai/"
EXPERIMENT_NAME = "[country] [city] [user] model + version"
