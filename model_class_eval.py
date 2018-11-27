
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

prefix = "my_class_"

print(tf.__version__)

df = pd.read_csv('query_data.csv')
train, test = train_test_split(df, test_size=0.2)
train_data = train[['attr1_l', 'attr1_u', 'attr2_l', 'attr2_u']].values
train_labels = train[['cnt']].values

test_data = test[['attr1_l', 'attr1_u', 'attr2_l', 'attr2_u']].values
test_labels = test[['cnt']].values
test_labels = test_labels > 0

print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

def build_model():
  model = keras.Sequential([
      keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(4,)),
      keras.layers.Dense(512, activation=tf.nn.sigmoid),
      keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])

  model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='binary_crossentropy',
              metrics=['accuracy'])
  return model

model = build_model()
model.load_weights('./checkpoints/' + prefix + 'checkpoint')

[loss, acc] = model.evaluate(test_data, test_labels, verbose=0)
predictions = model.predict(test_data)
print(predictions)
print()
print("Testing set Accuracy: {:7.2f}".format(acc))

with open(prefix + 'history.json') as f:
    history = json.load(f)

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(history['epoch'], np.array(history['acc']),
           label='Train Loss')
  plt.legend()

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

for bx in range(b):
  for by in range(b):
    lx = xmin + ((xmax - xmin) / b) * bx
    ux = xmin + ((xmax - xmin) / b) * (1 + bx)
    ly = ymin + ((ymax - ymin) / b) * by
    uy = ymin + ((ymax - ymin) / b) * (1 + by)
    cnt = df[(df['attr1'] >= lx) & (df['attr1'] < ux) & (df['attr2'] >= ly) & (df['attr2'] < uy)].count()[0]
    if cnt > 0:
      cnt = True
    else:
      cnt = False
    bins.append((lx, ux, ly, uy, cnt))

    a = np.array([lx, ux, ly, uy]).reshape((1,4))
    a = (a - mean) / std
    c = model.predict(a)
    if c[0][0] > 0.30:
      c = True
    else:
      c = False
    bins_est.append((lx, ux, ly, uy, c))


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False)
for bin in bins:
  if bin[4]:
    rect = patches.Rectangle((bin[0], bin[2]), bin[1] - bin[0], bin[3] - bin[2],linewidth=0,edgecolor='r',facecolor='black', alpha=1)
    ax1.add_patch(rect)

ax1.set_xlim(xmin,xmax)
ax1.set_ylim(ymin, ymax)


for bin in bins_est:
  if bin[4]:
    rect = patches.Rectangle((bin[0], bin[2]), bin[1] - bin[0], bin[3] - bin[2],linewidth=0,edgecolor='r',facecolor='black', alpha=1)
    ax2.add_patch(rect)

ax2.set_xlim(xmin,xmax)
ax2.set_ylim(ymin, ymax)

ax3.hist2d(a1, a2, bins=20, norm=LogNorm())

plt.show()