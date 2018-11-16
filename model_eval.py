
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
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

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.load_weights('./checkpoints/my_checkpoint')

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
predictions = model.predict(test_data)
print(predictions)
print()
print("Testing set Mean Abs Error: {:7.2f}".format(mae))

with open('history.json') as f:
    history = json.load(f)

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history['epoch'], np.array(history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history['epoch'], np.array(history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 200])

if True:

  plot_history(history)
  plt.show()


df = pd.read_csv("data_2d_corr.csv")
df = df[(df['attr1'] > 40) & (df['attr1'] < 60) ]
a1 = df['attr1']
a2 = df['attr2']
a1 = a1.values.reshape(len(a1),1)
a2 = a2.values.reshape(len(a2),1)

f1 = plt.figure()
ax1 = f1.add_subplot(111)

n, bins, patches = ax1.hist(a1, 20, facecolor='g', alpha=0.5)


# predict the histogram
counts = []
predicted_hist = []
for i in range(len(bins) - 1):
    lower = bins[i]
    upper = bins[i + 1]

    a = np.array([lower, upper, np.min(a2), np.max(a2)]).reshape((1,4))
    a = (a - mean) / std
    c = model.predict(a)[0][0]
    
    predicted_hist.append(lower)
    predicted_hist.append(upper)
    counts.append(c)
    counts.append(c)
    #print("count of " + str(lower) + " to " + str(upper) + " = " + str(c))
ax1.plot(predicted_hist, counts, color='blue', linewidth=1)

plt.show()