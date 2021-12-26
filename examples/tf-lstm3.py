
# 这是一个lstm的实验例子
# tensorflow 2.7.0 版本测试通过


import tensorflow as tf

class MyBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(MyBlock, self).__init__()
        self.lstm_1 = tf.keras.layers.LSTM(8,return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(5)
        self.dense_3 = tf.keras.layers.Dense(1,activation='sigmoid')

    def call(self, inputs):
        x = self.lstm_1(inputs)
        print(x)
        x = self.lstm_2(x)
        print(x)
        return self.dense_3(x)



import numpy as np


inputs = tf.keras.Input(shape=(32,5))
outputs =  MyBlock()(inputs)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Just use `fit` as usual
x = np.random.random((1000, 32,5))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=4)
