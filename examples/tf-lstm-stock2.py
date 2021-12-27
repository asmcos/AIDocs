
# 这是一个lstm的实验例子
# tensorflow 2.7.0 版本测试通过


import tensorflow as tf
import requests
import numpy as np
import pandas as pd
import json
import time


#
# config
#

NUM_EPOCHS = 5

#
# 1. stock data
#


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
y_train = []
def reshape_data(df, sequence_length=40):

    datas1 = df.loc[:,fields]

    if len(datas1) <= sequence_length:
        return

    for index in range(len(datas1) - sequence_length):
         X_train.append(datas1[index: index + sequence_length].values)
         y_train.append(datas1.values[index + sequence_length])
    return np.array(X_train),np.array(y_train)

inputs,labels = reshape_data(datas)
inputs = np.reshape(inputs,(inputs.shape[0],inputs.shape[1],len(fields)))
print(inputs)
#print(labels)


#
# 2. create model
#

class MyModel(tf.keras.models.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(32)
        self.d1 = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer="l2")
        self.d2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        # inputs: [batch_size, seq_len]

        x = self.lstm1(inputs)
        x = self.d1(x)  
        x = self.d2(x)  
        print(x)
        return x

model = MyModel()



#
# 3. train
#


optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.BinaryCrossentropy()
metrics = tf.keras.metrics.BinaryAccuracy()


@tf.function
def train_step(inputs, labels):
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    regularization_loss=tf.math.add_n(model.losses)
    pred_loss=loss_fn(labels, predictions)
    total_loss=pred_loss + regularization_loss

  gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 训练 train
for epoch in range(NUM_EPOCHS):
  train_step(inputs, labels)
  print("Finished epoch", epoch)

# 评估
logits = model(inputs)
metrics(labels, logits)
print(metrics.result())
