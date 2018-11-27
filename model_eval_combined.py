
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import sys

import json
from matplotlib.colors import LogNorm
import matplotlib.patches as patches

def norm(v, minv, maxv):
    if (v == 0):
        return 0
    n = (v - minv) / (maxv - minv)
    alpha = 0.05
    return (alpha + pow(n, 1.0 / 3.0) * (1.0 - alpha))

def load_model(prefix):
    if "class" in prefix:
        model = keras.Sequential([
            keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(4,)),
            keras.layers.Dense(512, activation=tf.nn.sigmoid),
            keras.layers.Dense(1, activation=tf.nn.sigmoid)
        ])

        model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='binary_crossentropy',
              metrics=['accuracy'])
        model.load_weights('./checkpoints/' + prefix + 'checkpoint')
        return model
    else:
        model = keras.Sequential([
          keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(4,)),
          keras.layers.Dense(256, activation=tf.sigmoid),
          keras.layers.Dense(512, activation=tf.nn.relu, kernel_constraint=keras.constraints.NonNeg()),
          keras.layers.Dense(1)
        ])

        model.compile(loss='mse',
              optimizer=tf.train.RMSPropOptimizer(0.001),
              metrics=['mae'])
        model.load_weights('./checkpoints/' + prefix + 'checkpoint')
        return model

def mean_std():
    pre_computed_df = pd.read_csv('query_data.csv')
    pre_computed_train_data = pre_computed_df[['attr1_l', 'attr1_u', 'attr2_l', 'attr2_u']].values
    mean = pre_computed_train_data.mean(axis=0)
    std = pre_computed_train_data.std(axis=0)
    return mean, std

def compute_gt(df, estimation_params, lx, ux, ly, uy):   
    cnt = df[(df['attr1'] >= lx) & (df['attr1'] < ux) & (df['attr2'] >= ly) & (df['attr2'] < uy)].count()[0]
    return cnt

def compute_combined_model(df, estimation_params, lx, ux, ly, uy):   
    a = np.array([lx, ux, ly, uy]).reshape((1,4))
    a = (a - estimation_params['mean']) / estimation_params['std']
    cnt = 0
    if estimation_params['classifier'] is not None: 
        cnt = estimation_params['classifier'].predict(a)
        if cnt[0][0] > estimation_params['classifier_threshold']:
          cnt = estimation_params['regressor'].predict(a)[0][0]
        else:
          cnt = 0
    else:
        cnt = estimation_params['regressor'].predict(a)[0][0]
    return cnt

def compute(df, estimation_params, nr_bins, fn):
    a1 = df['attr1']
    a2 = df['attr2']

    xmax = df['attr1'].max() + 0.05
    xmin = df['attr1'].min()
    ymax = df['attr2'].max() + 0.05
    ymin = df['attr2'].min()
    bins = []

    for bx in range(nr_bins):
      for by in range(nr_bins):
        lx = xmin + ((xmax - xmin) / nr_bins) * bx
        ux = xmin + ((xmax - xmin) / nr_bins) * (1 + bx)
        ly = ymin + ((ymax - ymin) / nr_bins) * by
        uy = ymin + ((ymax - ymin) / nr_bins) * (1 + by)
        cnt = fn(df, estimation_params, lx, ux, ly, uy)
        bins.append((lx, ux, ly, uy, cnt))
    return bins


def data_plot(bins, df, ax):
    min_cnt = min([ bin[4] for bin in bins ])
    max_cnt = max([ bin[4] for bin in bins ])
    
    for bin in bins:
        a = norm(bin[4], min_cnt, max_cnt)   
        rect = patches.Rectangle((bin[0], bin[2]), bin[1] - bin[0], bin[3] - bin[2], linewidth=0, edgecolor='r', facecolor='black', alpha=a)
        ax.add_patch(rect)

    xmax = df['attr1'].max()
    xmin = df['attr1'].min()
    ymax = df['attr2'].max()
    ymin = df['attr2'].min()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

def main():
    class_model_prefix = sys.argv[1]
    regr_model_prefix = sys.argv[2]
    nr_bins = int(sys.argv[3])


    class_model = None
    if class_model_prefix != 'none':
        class_model = load_model(class_model_prefix)
    regr_model = load_model(regr_model_prefix)

    mean, std = mean_std()
    estimation_params = {'classifier': class_model, 'regressor': regr_model, 'classifier_threshold': 0.01, 'mean': mean, 'std': std}

    df = pd.read_csv("data_2d_corr.csv")

    bins_gt = compute(df, estimation_params, nr_bins, compute_gt)
    bins_est = compute(df, estimation_params, nr_bins, compute_combined_model)
    errors = [ tup[0][4] - tup[1][4] for tup in zip(bins_gt, bins_est) ]

    print(np.mean(errors), sum([ abs(e) for e in errors ]))

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=False, figsize=(15,8))
    data_plot(bins_gt, df, ax1)
    data_plot(bins_est, df, ax2)
    ax3.hist2d(df['attr1'], df['attr2'], bins=nr_bins, norm=LogNorm())
    n, bins, patches = ax4.hist(errors, nr_bins, facecolor='g', alpha=0.5)
    plt.show()



if __name__ == '__main__':
    main()