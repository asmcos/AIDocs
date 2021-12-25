
# 这是一个lstm的实验例子
# tensorflow 2.7.0 版本测试通过


import tensorflow as tf


inputs = tf.random.normal([64,40,5])

# 第一步 进行lstm运算，保持 time_steps
lstm1 = tf.keras.layers.LSTM(8,return_sequences=True)
output1 = lstm1(inputs)
print(output1.shape)

# 第二步 return_sequences 使用默认值 False, 压缩time_steps
lstm2 = tf.keras.layers.LSTM(5)
output2 = lstm2(output1)

print(output2.shape)


sigmoid = tf.keras.layers.Dense(1,activation='sigmoid')(output2)
print(sigmoid.shape)


