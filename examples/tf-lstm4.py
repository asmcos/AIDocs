#Effective Tensorflow 2
#https://github.com/tensorflow/docs/blob/master/site/en/guide/effective_tf2.ipynb
#https://www.tensorflow.org/guide/effective_tf2
import tensorflow as tf
import numpy as np

BUFFER_SIZE = 10 # Use a much larger value for real code
BATCH_SIZE = 64
NUM_EPOCHS = 5


STEPS_PER_EPOCH = 5

class MyModel(tf.keras.models.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(32)
        self.d1 = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer="l2")
        self.d2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        # inputs: [batch_size, seq_len]

        x = self.lstm1(inputs)
        print(x.shape)
        x = self.d1(x)  # [batch_size, 16]
        x = self.d2(x)  # [batch_size, 1]]
        return x

model = MyModel()

optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.BinaryCrossentropy()

@tf.function
def train_step(inputs, labels):
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    regularization_loss=tf.math.add_n(model.losses)
    pred_loss=loss_fn(labels, predictions)
    total_loss=pred_loss + regularization_loss

  gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

x = np.random.random((1000, 32,10))
y = np.random.random((1000, 1))

for epoch in range(NUM_EPOCHS):
  train_step(x, y)
  print("Finished epoch", epoch)
