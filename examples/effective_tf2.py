#Effective Tensorflow 2
#https://github.com/tensorflow/docs/blob/master/site/en/guide/effective_tf2.ipynb
#https://www.tensorflow.org/guide/effective_tf2
import tensorflow as tf
import numpy as np

BUFFER_SIZE = 10 # Use a much larger value for real code
BATCH_SIZE = 64
NUM_EPOCHS = 5


STEPS_PER_EPOCH = 5


model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32,input_shape=(32,10),return_sequences=True),
    tf.keras.layers.Conv1D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.02),
                           input_shape=(32, 10)),

    tf.keras.layers.MaxPooling1D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(32)
])

optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(inputs, labels):
  print("train_step")
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    print(model.losses)
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
