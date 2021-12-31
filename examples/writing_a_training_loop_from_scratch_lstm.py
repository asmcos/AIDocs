#
# tensorflow documents
# https://tensorflow.google.cn/guide/keras/writing_a_training_loop_from_scratch?hl=zh-cn
#

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras.utils import np_utils
# 如果下载数据ssl验证不过，可以添加下面2行
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


#
#  1. prepare data
#
# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# 原始数据 60000，28，28

#
# for lstm 转化数据类型
#
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.


# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)



#
# 2. model
# my Model test lstm
#

class MyModel(tf.keras.models.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(32,input_shape=(28,28))
        self.d1 = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer="l2")
        self.d2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=None, mask=None):
        # inputs: [batch_size, seq_len]
        x = self.lstm1(inputs)
        x = self.d1(x)  # [batch_size, 16]
        x = self.d2(x)  # [batch_size, 1]]
        return x


model = MyModel()

# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


#
# 3. train 2 第二种写法
#  训练和评估
#

train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

import time

epochs = 2
start_time = time.time()
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))


# Reset training metrics at the end of each epoch
train_acc_metric.reset_states()

# Run a validation loop at the end of each epoch.
for x_batch_val, y_batch_val in val_dataset:
     val_logits = model(x_batch_val, training=False)
     # Update val metrics
     val_acc_metric.update_state(y_batch_val, val_logits)
    
val_acc = val_acc_metric.result()
val_acc_metric.reset_states()
print("Validation acc: %.4f" % (float(val_acc),))
print("Time taken: %.2fs" % (time.time() - start_time))



