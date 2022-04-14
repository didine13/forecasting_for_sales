import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_log_error
import joblib
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

data_path = 'raw_data/'
#data = pd.read_csv(data_path + "train2016.csv")


def get_setdata(path, file, **kwargs):
  nrows=kwargs.get('nrows', None)

  if nrows:
    raw_data = pd.read_csv(path + file, nrows=nrows)
  else:
    raw_data = pd.read_csv(path + file)
  print(f'Chargement du fichier {file} OK')
  return raw_data

def encoder(data, encoder="ohe",):
  categorical_data = data.select_dtypes(include=["bool","object"]).columns
  for i in categorical_data:
    print(i)
    # Boolean colums
    if len(data[i].unique()) == 2:
      ohe = OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse = False)
      data[i] = ohe.fit_transform(data[[i]])
    # Non boolean columns
    else:
      ohe = OneHotEncoder()
      data = data.join(pd.DataFrame(ohe.fit_transform(data[[i]]).toarray(), columns=ohe.get_feature_names()))
      data = data.drop(columns=i)

    #  data = pd.concat([data, ohe_data], axis=1).drop(i, axis=1)

  print("OHE ended")
  return data


def RMSLE(y_true:np.ndarray, y_pred:np.ndarray) -> np.float64:
  return np.sqrt(mean_squared_log_error(y_true, y_pred))

def MAPE(y_true:np.ndarray, y_pred:np.ndarray) -> np.float64:
  pass


def replace_by_unit_sales(row):
  val = row['family_sales']
  for i in range(5, len(row)):
    if row[i] != 0.0:
      row[i] = val
  return row


data = get_setdata(data_path,"preprocessed_sales_grouped_1.csv")
data['date']=pd.to_datetime(data['date'])

if len(data.store_nbr.unique() == 1):
  our_store = data.store_nbr.unique()[0]
else:
  print('Pb with store number')

oil = get_setdata(data_path, 'oil.csv')
# replace nan by modal value
oil['dcoilwtico'].replace([np.nan], oil['dcoilwtico'].mode()[0], inplace=True)

# scale minmax scaler
scaler = MinMaxScaler()
oil['dcoilwtico'] = scaler.fit_transform(oil[['dcoilwtico']])

# to_datetime on date
oil['date'] = pd.to_datetime(oil['date'])

store_test = data.copy()
store_test = encoder(store_test)

# for each day, put unit_sales below each family
store_test2 = store_test.apply(lambda row: replace_by_unit_sales(row), axis=1)
store_test2 = store_test2.groupby('date').sum().reset_index()
store_test2 = store_test2.merge(oil, on='date')
store_test2.set_index('date', inplace=True)
store_test2.sort_index(inplace=True)
store_test2["store_nbr"] = our_store
store_test2["is_special"] = np.where(store_test2['is_special'] >= 2, 1, store_test2['is_special'])
store_test2["is_open"] = np.where(store_test2['is_open'] >= 2, 1, store_test2['is_open'])


#pip install pmdarima --quiet
#pip install --quiet statsmodels==0.11

colonnes = list(store_test2.columns)
colonnes.remove("store_nbr")
colonnes.remove("family_sales")
done = ['is_open', 'is_special',"x0_BREAD/BAKERY"]
for i in colonnes:

    if i not in done and i != "dcoilwtico":
        liste_temp = colonnes.copy()

        endo = i
        liste_temp.remove(i)
        exo = liste_temp.copy()
        done.append(i)
        print(f"endo : {endo} -- exo = {exo}")


        index = round(0.75 * store_test2.shape[0])
        train = store_test2.iloc[0:index]
        test = store_test2.iloc[index+1:]
        sarimax = pm.auto_arima(
                      store_test2[[endo]],
                      X=store_test2[exo].values,
                      start_p=0, start_q=0,
                      test='adf',
                      max_p=2, max_q=2, m=12,
                      start_P=0, seasonal=True,
                      d=None, D=1, trace=True,
                      error_action='warn',
                      suppress_warnings=True,
                      n_jobs= -1,
                      stepwise=True)

        # nom du modèle à exporter
        model_name = f"model_{our_store}_{endo}.joblib"
        model_name = data_path+model_name.replace('/','_')

        joblib.dump(sarimax, model_name)
