
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

import json
from matplotlib.colors import LogNorm
import matplotlib.patches as patches

prefix = "gcp_3_"

print(tf.__version__)

def norm(v, minv, maxv):
  if (v == 0):
    return 0
  n = (v - minv) / (maxv - minv)
  alpha = 0.05
  return (alpha + pow(n, 1.0 / 3.0) * (1.0 - alpha))

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
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(4,)),
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
model.load_weights('./checkpoints/' + prefix + 'checkpoint')

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
predictions = model.predict(test_data)
print(predictions)
print()
print("Testing set Mean Abs Error: {:7.2f}".format(mae))

with open(prefix + 'history.json') as f:
    history = json.load(f)

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(history['epoch'], np.array(history['mean_absolute_error']),
           label='Train Loss')
  #plt.plot(history['epoch'], np.array(history['val_mean_absolute_error']),
  #         label = 'Val loss')
  plt.legend()
  #plt.ylim([0, 75])

if True:
  plot_history(history)
  plt.show()


df = pd.read_csv("data_2d_corr.csv")
#df = df[(df['attr1'] > 55) & (df['attr1'] < 60) ]
a1 = df['attr1']
a2 = df['attr2']

xmax = df['attr1'].max()
xmin = df['attr1'].min()
ymax = df['attr2'].max()
ymin = df['attr2'].min()
b = 20
bins = []
bins_est = []

max_cnt = float('-inf')
min_cnt = float('inf')

max_est_cnt = float('-inf')
min_est_cnt = float('inf')

trues = []
errors = []
percent = []
for bx in range(b):
  for by in range(b):
    lx = xmin + ((xmax - xmin) / b) * bx
    ux = xmin + ((xmax - xmin) / b) * (1 + bx)
    ly = ymin + ((ymax - ymin) / b) * by
    uy = ymin + ((ymax - ymin) / b) * (1 + by)
    cnt = df[(df['attr1'] >= lx) & (df['attr1'] < ux) & (df['attr2'] >= ly) & (df['attr2'] < uy)].count()[0]
    bins.append((lx, ux, ly, uy, cnt))
    max_cnt = max(max_cnt, cnt)
    min_cnt = min(min_cnt, cnt)

    a = np.array([lx, ux, ly, uy]).reshape((1,4))
    a = (a - mean) / std
    c = model.predict(a)[0][0]
    bins_est.append((lx, ux, ly, uy, c))

    max_est_cnt = max(max_est_cnt, c)
    min_est_cnt = min(min_est_cnt, c)

    trues.append(cnt)
    errors.append(abs(cnt - c))
    if cnt != 0:
      percent.append(abs(cnt - c) / cnt)

print('percent', np.mean(percent))
print('true cnt', np.mean(trues))
print('abs mean error', np.mean(errors))
print("real", min_cnt, max_cnt)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False)
for bin in bins:
  a = norm(bin[4], min_cnt, max_cnt)   
  rect = patches.Rectangle((bin[0], bin[2]), bin[1] - bin[0], bin[3] - bin[2],linewidth=0,edgecolor='r',facecolor='black', alpha=a)
  ax1.add_patch(rect)

ax1.set_xlim(xmin,xmax)
ax1.set_ylim(ymin, ymax)


print("est", min_est_cnt, max_est_cnt)
for bin in bins_est:
  a = norm(bin[4], min_est_cnt, max_est_cnt) 
  rect = patches.Rectangle((bin[0], bin[2]), bin[1] - bin[0], bin[3] - bin[2],linewidth=0,edgecolor='r',facecolor='black', alpha=a)
  ax2.add_patch(rect)

ax2.set_xlim(xmin,xmax)
ax2.set_ylim(ymin, ymax)

ax3.hist2d(a1, a2, bins=20, norm=LogNorm())

plt.show()

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