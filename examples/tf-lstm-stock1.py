
# 这是一个lstm的实验例子
# tensorflow 2.7.0 版本测试通过


import tensorflow as tf
import requests
import numpy as np
import pandas as pd
import json
import time

hostname="http://klang.org.cn"
def get_stock(code,start='2021-01-01',end='2021-12-20'):
    try:
        json = requests.get(hostname+"/dayks",
            params={"code":code,"end":end,"limit":200},timeout=1000).json()
    except:
        time.sleep(2)
        json = requests.get(hostname+"/dayks",
            params={"code":code,"end":end,"limit":200},timeout=1000).json()

    df = pd.json_normalize(json)
    if len(df) < 1:
       return []
    df = df.drop(columns=['_id','codedate','id'])
    datas = df.sort_values(by="date",ascending=True)

    return datas

datas = get_stock('sh.600600')

fields = ['close']

X_train = []
def reshape_data(df, sequence_length=40):

    datas1 = df.loc[:,fields]

    if len(datas1) <= sequence_length:
        return

    for index in range(len(datas1) - sequence_length):
         X_train.append(datas1[index: index + sequence_length].values)

    return np.array(X_train)

inputs = reshape_data(datas)
inputs = np.reshape(inputs,(inputs.shape[0],inputs.shape[1],len(fields)))
print(inputs)

lstm1 = tf.keras.layers.LSTM(1) #,return_sequences=True)
output1 = lstm1(inputs)
print(output1.shape)
print(output1)

