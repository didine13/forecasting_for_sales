import joblib
from google.cloud import storage
from forecasting_for_sales.data import *
from forecasting_for_sales.params import *
from termcolor import colored
import os
from forecasting_for_sales.utils import *


@simple_time_tracker
def get_data_from_gcp(nrows=10000, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    path = f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}"
    df = pd.read_csv(path, nrows=nrows)
    return df



def train_model(X_train, y_train):
    """method that trains the model"""
    '''rgs = linear_model.Lasso(alpha=0.1)
    rgs.fit(X_train, y_train)
    print("trained model")
    return rgs'''
    pass


def storage_upload_to_gcp(rm=False):
    client = storage.Client().bucket(BUCKET_NAME)

    local_model_name = 'model.joblib'
    storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.joblib')

    print(colored(f"=> model.joblib uploaded to gcp cloud storage under, to bucket {BUCKET_NAME} inside {storage_location}",
                  "green"))
    if rm:
        os.remove('model.joblib')


def save_model_locally(reg):
    """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
    HINTS : use joblib library and google-cloud-storage"""

    joblib.dump(reg, 'model.joblib')
    print("saved model.joblib locally")


if __name__ == '__main__':
    # get training data from GCP bucket
    df = get_data_from_gcp()

    # preprocess data
    #X_train, y_train = preprocess(df)

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    reg = train_model(X_train, y_train)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model_locally(reg)

    # upload model on gcp
    storage_upload_to_gcp()
