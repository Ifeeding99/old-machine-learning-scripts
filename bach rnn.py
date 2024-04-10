import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import timeseries_dataset_from_array

dir_path = 'C:/Users/User01/Desktop/ambiente pycharm/venv/Scripts/programmi python/databases/jsb_chorales/train'
l = os.listdir(dir_path)
train_df = []
for f in l:
    file_path = dir_path + '/' + f
    arr = pd.read_csv(file_path).values.tolist()
    train_df.append(arr)

X = []
Y = []
n_values_to_predict = 20
print(train_df[0])
for el in train_df:
    X.append(el[:-1])
    Y.append(el[-1])
print(Y)




