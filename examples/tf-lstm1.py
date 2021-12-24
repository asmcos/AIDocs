
# 这是一个lstm的实验例子
# tensorflow 2.7.0 版本测试通过


import tensorflow as tf


inputs = tf.random.normal([64,40,5])

lstm = tf.keras.layers.LSTM(8)

output = lstm(inputs)
print(output.shape)

