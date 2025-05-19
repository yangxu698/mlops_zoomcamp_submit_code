!python -V
# get_ipython().system('python -V')

#%%
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

import glob, os
from pathlib import Path
cwd = os.getcwd()
ROOT_DIR = str(Path(cwd).parent)
#%%
df = pd.read_parquet(f"{ROOT_DIR}/Week1/raw_data/yellow_tripdata_2023-01.parquet")

df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

df.describe()

df= df.loc[(df.duration >= 1) & (df.duration <= 60)]

categorical = ['PULocationID', 'DOLocationID']
numerical = ['trip_distance']

df[categorical] = df[categorical].astype(str)

#%%
train_dicts = df[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

target = 'duration'
y_train = df[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

root_mean_squared_error(y_train, y_pred)

#%%
sns.displot(y_pred, label='prediction')
sns.displot(y_train, label='actual')
plt.legend()

#%%
def read_dataframe(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)

        df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
        df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df.loc[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

#%%
df_train = read_dataframe(f'{ROOT_DIR}/Week1/raw_data/yellow_tripdata_2023-01.parquet')
df_val = read_dataframe(f'{ROOT_DIR}/Week1/raw_data/yellow_tripdata_2023-02.parquet')

#%%
len(df_train), len(df_val)

#%%
df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']

#%%
categorical = ['PULocationID', 'DOLocationID'] # 'PU_DO' 
numerical = ['trip_distance']

dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

#%%
target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values

#%%
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_val)
y_train_pred = lr.predict(X_train)

root_mean_squared_error(y_train, y_train_pred)
root_mean_squared_error(y_val, y_pred)

#%%
with open(f'{ROOT_DIR}/Week1/models/lin_reg.bin', 'wb') as f_out:
    pickle.dump((dv, lr), f_out)

#%%
# lr = Lasso(0.01)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_val)
root_mean_squared_error(y_val, y_pred)
