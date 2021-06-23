import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import scipy.stats
from scipy import signal
from ast import literal_eval
import torch
from train_GRUD.py import *
from GRUD.py import *

from sklearn.model_selection import train_test_split

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

pd.set_option('display.max_columns', None)

DATAPATH = "~/datasets/stressrecov"

EXTENSION = "/oura"

sleep_df = pd.read_parquet("~/stressrecovery/data/processed/oura/sleep_concat.parquet")

sleep_df['date']=sleep_df['date'].astype(str) 

daily_merged = pd.read_parquet("~/stressrecovery/data/processed/survey/daily_merged.parquet")
daily_merged['date']=daily_merged['date'].astype(str) 

daily_merged["covid_shift_any"] = daily_merged.apply (lambda row: (row.daily_covid_shifts___1 or row.daily_covid_shifts___2 or row.daily_covid_shifts___3 ), axis=1)

df = pd.merge(sleep_df, daily_merged,  how='left', on=["participant_id","date"])
df['date'] = df["date"].astype(str)
df['date'] = pd.to_datetime(df["date"])

df.sort_values(['participant_id', 'date'], ascending=True)


#df = df.dropna(subset = ["daily_shifts", "daily_covid_shifts___1", "daily_covid_shifts___2", "daily_covid_shifts___3"])
df =df.dropna(subset=["hr_5min","rmssd_5min"])
df["shift_any"] = df.apply (lambda row: (row.daily_shifts ==1.0 or row.daily_shifts==2.0), axis=1)
df["covid_shift_any"] = df["covid_shift_any"].astype(bool)

def covidshift(row):
    if row["covid_shift_any"] and row["shift_any"]:
        return True
    elif row["shift_any"] == False:
        return None
    else:
        return False


def array_preprocess_DROP(arr):
    dropped = arr[arr!=0]
    if len(dropped)!=0:
        return arr[arr!=0]
    else:
        return None

df["covidshift"] = df.apply (lambda row: covidshift(row), axis=1)

#Turn to literal arrays and drop 0s (missing values)
df["hr_5min"] = df.apply (lambda row: array_preprocess_DROP(np.array(literal_eval(row.hr_5min))), axis=1)
df["rmssd_5min"] = df.apply (lambda row: array_preprocess_DROP(np.array(literal_eval(row.rmssd_5min))), axis=1)

#DROPPING NANs AGAIN BECAUSE PREVIOUS OPERATION WOULD HAVE GENERATED NANs ie: array of [0,0,0] -> NaN
df =df.dropna(subset=["hr_5min","rmssd_5min"])

df["rmssd_lowest"] = df.apply (lambda row: np.amin(row.rmssd_5min), axis=1)

df["hr_max"] = df.apply (lambda row: np.amax(row.hr_5min), axis=1)
df["rmssd_max"] = df.apply (lambda row: np.amax(row.rmssd_5min), axis=1)

df["hr_1quantile"] = df.apply (lambda row: np.quantile(row.hr_5min, 0.25), axis=1)
df["rmssd_1quantile"] = df.apply (lambda row: np.quantile(row.rmssd_5min, 0.25), axis=1)

df["hr_2quantile"] = df.apply (lambda row: np.quantile(row.hr_5min, 0.50), axis=1)
df["rmssd_2quantile"] = df.apply (lambda row: np.quantile(row.rmssd_5min, 0.50), axis=1)

df["hr_3quantile"] = df.apply (lambda row: np.quantile(row.hr_5min, 0.75), axis=1)
df["rmssd_3quantile"] = df.apply (lambda row: np.quantile(row.rmssd_5min, 0.75), axis=1)

#Columns of data we want to keep (contains input and output, output is dropped later)

all_columns = ["awake","breath_average", "deep", "duration", "hr_average", "hr_lowest",
          "light", "onset_latency", "rem", "restless", "rmssd", "temperature_delta", "temperature_trend_deviation",
          "total", "daily_stressed", "daily_shifts", "daily_control", "daily_reduce", "rmssd_lowest","hr_max",
           "rmssd_max", "hr_1quantile", "rmssd_1quantile", "hr_2quantile", "rmssd_2quantile", "hr_3quantile",
          "rmssd_3quantile", "score", "score_bin_0", "score_bin_1", "score_bin_2", "score_bin_3", "score_bin_4"]
#Columns to keep (date, participant_id, daily_stressed dropped post processing)
columns = ["date","participant_id", "deep", "rmssd","hr_average","score_bin_0", "score_bin_1", "score_bin_2", "score_bin_3", "score_bin_4"]
df_clean = df.copy()
df = df[['date', 'participant_id','rmssd','hr_average','daily_stressed']]

# Load Participant Train/Val/Test Split dictionary
read_dictionary = np.load('participant_splits.npy',allow_pickle='TRUE').item()

train_df = df[df["participant_id"].isin(read_dictionary["train"])]
val_df = df[df["participant_id"].isin(read_dictionary["val"])]
test_df = df[df["participant_id"].isin(read_dictionary["test"])]

#groud by participant 
test_df = test_df[test_df.participant_id == 'U-1GPMFTVN6JZLK4G5UKFA']

#create empty rows for missing dates
test_df = test_df.set_index('date')
test_df = test_df.resample('D').mean()

#ffill label
test_df['daily_stressed'] = test_df['daily_stressed'].fillna(method='ffill')

#masking matrix
stress = test_df['daily_stressed']
test_df =  test_df.drop(["daily_stressed"], axis=1)


# forward fill (not the one used for creating the mask array)
test_df_imp = test_df.copy()
test_df_imp = test_df_imp.ffill(axis = 0)

time_len = test_df.shape[0]
seq_len = 10
pred_len = 1
sequences, labels, sequences_tmp = [], [], []

for i in range(time_len - seq_len - pred_len):
    sequences.append(test_df_imp.iloc[i:i+seq_len].values)
    sequences_tmp.append(test_df.iloc[i:i+seq_len].values)
    labels.append(stress.iloc[i+seq_len:i+seq_len+pred_len].values)

b = np.copy(sequences_tmp)
b = np.where(b >= 0, 1, b)
Mask = np.nan_to_num(b)

interval = 1
S = np.zeros_like(sequences) # time stamps
for i in range(S.shape[1]):
    S[:,i,:] = interval * i

Delta = np.zeros_like(sequences)


for i in range(1, S.shape[1]):
    Delta[:,i,:] = S[:,i,:] - S[:,i-1,:]


missing_index = np.where(Mask == 0)
X_last_obsv = np.copy(sequences)
for idx in range(missing_index[0].shape[0]):
  i = missing_index[0][idx] 
  j = missing_index[1][idx]
  k = missing_index[2][idx]
  if j != 0 and j != seq_len-1:
    Delta[i,j+1,k] = Delta[i,j+1,k] + Delta[i,j,k]
  if j != 0:
    X_last_obsv[i,j,k] = X_last_obsv[i,j-1,k] # last observation

Delta = Delta / Delta.max() # normalize

sequences = np.array(sequences)
sample_size = sequences.shape[0]
index = np.arange(sample_size, dtype = int)

sequences = sequences[index]
labels = np.array(labels)[index]


X_last_obsv = X_last_obsv[index]
Mask = Mask[index]
Delta = Delta[index]
sequences = np.expand_dims(sequences, axis=1)
X_last_obsv = np.expand_dims(X_last_obsv, axis=1)
Mask = np.expand_dims(Mask, axis=1)
Delta = np.expand_dims(Delta, axis=1)

dataset_agger = np.concatenate((sequences, X_last_obsv, Mask, Delta), axis = 1)

train_propotion = 0.8
valid_propotion = 0.2
train_index = int(np.floor(sample_size * train_propotion))
valid_index = int(np.floor(sample_size * ( train_propotion + valid_propotion)))

train_data, train_label = dataset_agger[:train_index], labels[:train_index]
valid_data, valid_label = dataset_agger[train_index:valid_index], labels[train_index:valid_index]
test_data, test_label = dataset_agger[valid_index:], labels[valid_index:]

train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

train_dataset = utils.TensorDataset(train_data, train_label)
valid_dataset = utils.TensorDataset(valid_data, valid_label)
test_dataset = utils.TensorDataset(test_data, test_label)
train_dataloader = utils.DataLoader(train_dataset, shuffle=False, drop_last = True)
valid_dataloader = utils.DataLoader(valid_dataset, shuffle=False, drop_last = True)
test_dataloader = utils.DataLoader(test_dataset, shuffle=False, drop_last = True)


X_mean = np.mean(sequences, axis = 0)

inputs, labels = next(iter(train_dataloader))

print(inputs.shape, labels.shape)
[batch_size, type_size, step_size, fea_size] = inputs.size()

input_dim = fea_size
hidden_dim = fea_size
output_dim = fea_size


grud = GRUD(input_dim, hidden_dim, output_dim, X_mean, output_last = True)

best_grud, losses_grud = Train_Model(grud, train_dataloader, valid_dataloader)

