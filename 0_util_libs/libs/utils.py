import pandas as pd
from scipy import stats
import datetime
import numpy as np 
from sklearn import preprocessing


def normalize_dataset(meu_data_frame):
    #instanciando o normalizador MinMAx
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    #obtendo as colunas do dataframe
    columns = meu_data_frame.columns

    #fazendo uma copia do dataframe, iremos normalizar a copia
    normalized_data_frame = meu_data_frame.copy()

    #iterando sobre cada coluna
    for column in columns:
        #verificando se a coluna é numérica
        if(meu_data_frame[column].dtype == "int64" or meu_data_frame[column].dtype == "float64"):
            x = meu_data_frame[column].values
            x_norm = min_max_scaler.fit_transform(x.reshape(-1, 1))
            normalized_data_frame[column] = pd.DataFrame(x_norm)
            normalized_data_frame.rename(columns={column:column+"_norm"}, inplace=True)

    return normalized_data_frame


def standarlize_dataset(meu_data_frame):
    standard_scaler = preprocessing.StandardScaler()
    columns = meu_data_frame.columns

    standarlized_data_frame = meu_data_frame.copy()

    for column in columns:
        if(meu_data_frame[column].dtype == "int64" or meu_data_frame[column].dtype == "float64"):
            x = meu_data_frame[column].values
            x_norm = standard_scaler.fit_transform(x.reshape(-1, 1))
            standarlized_data_frame[column] = pd.DataFrame(x_norm)
            standarlized_data_frame.rename(columns={column:column+"_stda"}, inplace=True)
    
    return standarlized_data_frame 