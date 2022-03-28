from cProfile import label
import tensorflow as tf
from unicodedata import name
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from pandas.core.frame import DataFrame
import argparse

def create_data(data, past , future ):
    x_train = []
    y_train = []

    for i in range(past,len(data)-future):
        t = data[i-past:i]
        y = data[i:i+future]
        x_train.append(t)
        y_train.append(y)
        

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train

def splitData(X,Y):
  X_train = X[:-15]
  Y_train = Y[:-15]
  X_val = X[-15:]
  Y_val = Y[-15:]
  return X_train, Y_train, X_val, Y_val

def buildManyToManyModel(shape):
  regressor = Sequential()
  regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (shape[1], shape[2])))
  regressor.add(Dropout(0.2))
  regressor.add(LSTM(units = 50, return_sequences = True))
  regressor.add(Dropout(0.2))
  regressor.add(LSTM(units = 50, return_sequences = True))
  regressor.add(Dropout(0.2))
  regressor.add(LSTM(units = 50))
  regressor.add(Dropout(0.2))
  regressor.add(Dense(units = 15))
  regressor.compile(loss="mse", optimizer="adam")
  regressor.summary()
  return regressor

if __name__ == '__main__':
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    data=pd.read_csv(args.training)
    data=data[['備轉容量(萬瓩)']]
    
    data_val=data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    x_train, y_train = create_data(data,15, 15)    
    x_train, y_train, x_test, y_test = splitData(x_train,y_train)

regressor= buildManyToManyModel(x_train.shape)
regressor.fit(x_train, y_train, epochs = 180 )

predict_y = regressor.predict(x_test)
predict_y = scaler.inverse_transform(predict_y)
# print(predict_y, np.array(data_val[-15:]))
# plt.plot(np.array(data_val[-15:]), 'b', label='actual')
# plt.plot(predict_y[-1][:] , 'r', label='predict')
# data_val=np.array(data_val[-15:])

rmse=0
rmse = mean_squared_error(data_val[-15:], predict_y[-1][:], squared=False)
print(rmse)

date=[n for n in range(20220330,20220332)]
date.extend([n for n in range(20220401,20220414)])
submission=pd.DataFrame({
        'date': date,
        'operating_reserve(MW)': predict_y[-1][:]*10
    })

submission.to_csv('submission.csv',index=False)