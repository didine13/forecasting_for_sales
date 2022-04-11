# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* forecasting_for_sales/*.py

black:
	@black scripts/* forecasting_for_sales/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr forecasting_for_sales-*.dist-info
	@rm -fr forecasting_for_sales.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


# ----------------------------------
#      CREATE BUCKET ON GCP
# ----------------------------------
# project id - replace with your GCP project id
PROJECT_ID=wagon-bootcam-341311

# bucket name - replace with your GCP bucket name (without capital letters)
BUCKET_NAME=wagon-data-835-forecasting-for-sales

##### Training  - - - - - - - - - - - - - - - - - - - - - -
# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = 'trainings'

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west1
PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=1.15
JOB_NAME=foracasting_for_sales_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')






# ----------------------------------
#      CREATE BUCKET ON GCP
# ----------------------------------

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}


# ----------------------------------
#      UPLOAD DATASET
# ----------------------------------

LOCAL_PATH=~/code/didine13/forecasting_for_sales/raw_data/train.csv

# bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
BUCKET_FOLDER=data

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

upload_data:
	# @gsutil cp train.csv gs://wagon-ml-my-bucket-name/data/train.csv
	@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}


# ----------------------------------
#      RUN MODEL LOCALLY
# ----------------------------------

PACKAGE_NAME=forecasting_for_sales
FILENAME=data_gcp

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}


# ----------------------------------
#      RUN MODEL ON GCP
# ----------------------------------


gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs


# ----------------------------------
#      API FOR PREDICTION
# ----------------------------------
run_api:
	uvicorn api.fast:app --reload  # load web server with code autoreload : web server pour l'appel de l'API



# ----------------------------------
#         HEROKU COMMANDS
# ----------------------------------

streamlit:
	-@streamlit run app.py

heroku_login:
	-@heroku login

heroku_create_app:
	-@heroku create ${APP_NAME}

# deploy_heroku:
# 	-@git push heroku master
# 	-@heroku ps:scale web=1
