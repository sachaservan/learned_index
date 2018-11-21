
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import json

print(tf.__version__)

df = pd.read_csv('query_data.csv')
train, test = train_test_split(df, test_size=0.2)
train_data = train[['attr1_l', 'attr1_u', 'attr2_l', 'attr2_u']].values
train_labels = train[['cnt']].values

test_data = test[['attr1_l', 'attr1_u', 'attr2_l', 'attr2_u']].values
test_labels = test[['cnt']].values


print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features

print(train_data[0])  # Display sample features, notice the different scales

# Test data is *not* used when calculating the mean and std

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])  # First training sample, normalized

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
    keras.layers.Dense(256, activation=tf.sigmoid),
    keras.layers.Dense(512, activation=tf.nn.relu, kernel_constraint=keras.constraints.NonNeg()),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def __init__(self, model):
    self.model = model
    self.history = {'epoch': [], 'val_loss' : [], 'val_mean_absolute_error' : [], 'loss' : [], 'mean_absolute_error' : []}
  def on_epoch_end(self, epoch, logs):
    for k in logs:
      self.history[k].append(logs[k])
    self.history['epoch'].append(epoch)

    with open('history.json', 'w') as outfile:
      json.dump(self.history, outfile, indent = 4)
    if epoch % 10 == 0:
      [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
      print("Testing set Mean Abs Error: {:7.2f}".format(mae))
      model.save_weights('./checkpoints/my_checkpoint')

EPOCHS = 2000

# Store training stats
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.0, verbose=1,
                    callbacks=[PrintDot(model)])


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()

plot_history(history)
plt.show()

