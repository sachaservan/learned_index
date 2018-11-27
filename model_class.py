
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import json
import random

prefix = "gcp_4_class_"

def gen_data(mean, std):
  df = pd.read_csv('data_2d_corr.csv')
  attr1_bounds = [df['attr1'].min(), df['attr1'].max()]
  attr2_bounds = [df['attr2'].min(), df['attr2'].max()]
  while 1: 
    table = []
    labels = []
    for i in range(10):
      attr1_filter = sorted([random.uniform(attr1_bounds[0], attr1_bounds[1]), random.uniform(attr1_bounds[0], attr1_bounds[1])])
      attr2_filter = sorted([random.uniform(attr2_bounds[0], attr2_bounds[1]), random.uniform(attr2_bounds[0], attr2_bounds[1])])
      cnt = df[(df['attr1'] >= attr1_filter[0]) & (df['attr1'] < attr1_filter[1]) & (df['attr2'] >= attr2_filter[0]) & (df['attr2'] < attr2_filter[1])].count()

      a = [attr1_filter[0], attr1_filter[1], attr2_filter[0], attr2_filter[1]]
      a = (a - mean) / std
      table.append(a)
      labels.append(cnt[0] > 0)
    yield np.array(table), np.array(labels)

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

class KerasCallback(keras.callbacks.Callback):
  def __init__(self, model):
    self.model = model
    self.history = {'epoch': [], 'loss' : [], 'acc' : []}
  def on_epoch_end(self, epoch, logs):
    for k in logs:
      self.history[k].append(logs[k])
    self.history['epoch'].append(epoch)

    with open(prefix + 'history.json', 'w') as outfile:
      json.dump(self.history, outfile, indent = 4)
    if epoch % 10 == 0:
      model.save_weights('./checkpoints/' + prefix + 'checkpoint')

pre_computed_df = pd.read_csv('query_data.csv')
pre_computed_train_data = pre_computed_df[['attr1_l', 'attr1_u', 'attr2_l', 'attr2_u']].values
mean = pre_computed_train_data.mean(axis=0)
std = pre_computed_train_data.std(axis=0)

print("dummy percentage", pre_computed_df[pre_computed_df['cnt'] > 0].count()[0] /len(pre_computed_df))

EPOCHS = 5000
STEPS_PER_EPOCH = 1000
model = build_model()
model.summary()

history = model.fit_generator(gen_data(mean, std), epochs=EPOCHS,
                    verbose=1, steps_per_epoch=STEPS_PER_EPOCH,
                    workers=1, use_multiprocessing=False,
                    callbacks=[KerasCallback(model)])

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(history['epoch'], np.array(history['acc']),
           label='Train Loss')
  plt.legend()

plot_history(history)
plt.show()

