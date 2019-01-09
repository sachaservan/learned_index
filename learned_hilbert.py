import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from optparse import OptionParser
from hilbert import HilbertCurve
from math import floor
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
# read csv -> points
# TODO determine size of HilbertCurve
#   maximum point is 2**p-1
#   minimum point is 0
#   all points have to be integers
#

DIM = 2
NUM_P = 2**10  # 32k


def generate_data(dimension=DIM, num_points=NUM_P):
    max_val = 2**int(np.floor(np.log2(num_points) / 2))
    print("max val", max_val)
    points = np.random.rand(num_points, dimension) * max_val
    df = pd.DataFrame(points, columns=['x', 'y'])
    df.index.name = 'index'
    path = os.path.abspath(f'./data_{dimension}d.csv')
    df.to_csv(path)
    return path


def main():
    path = generate_data()
    df = pd.read_csv(path, index_col=['index'])
    # num_d = len(df.columns); for i in range(num_d)...
    x = df.iloc[:, [0]]
    y = df.iloc[:, [1]]
    xf = np.floor(x).astype(int)
    yf = np.floor(y).astype(int)

    hp = np.ceil(np.floor(np.log2(NUM_P)) / 2.0).astype(int)
    curve = HilbertCurve(
        p=hp, n=DIM)

    plist = np.reshape(np.dstack((xf, yf)), (-1, DIM))

    max_num_points = 2**(hp * 2)
    out_array = [None] * len(plist)

    print("max points", max_num_points)

    for i, p in enumerate(plist):
        ind = curve.distance_from_coordinates(p)
        out_array[i] = (ind, np.asscalar(x.iloc[i]), np.asscalar(y.iloc[i]))

    out_array.sort(key=lambda x: x[0])
    sns.scatterplot(x=[i[1] for i in out_array],
                    y=[i[2] for i in out_array],
                    hue=[i[0] for i in out_array])
    plt.show()

    n_array = np.array([[x, y] for _, x, y in out_array])

    nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute').fit(n_array)
    distances, indices = nbrs.kneighbors(n_array)

    # print(distances[:50])
    print(indices[:50])

    return out_array


class Point:

    point_count = 0

    def __init__(self):
        self.point_count = Point.point_count
        Point.point_count += 1


class Node:
    """Represents a line.

    Children are finer grained Node(s).
    Max and Min are for ranges.
    m and b come from the equation y = mx + b
    """

    node_count = 0

    def __init__(self):
        Node.node_count += 1
        self.m = 0
        self.b = 0
        self.children = []
        self.max = 0
        self.min = 0


def build_recursive(X, Y, w, d, current_d):
    """Recursively build Node datastructure.

    Args:
        X (np.Array): X values
        Y (np.Array): Y values
        w (int): the number of buckets to create
        d (int): the number of levels of recursion
        current_d (int): current depth (usage is to call this with 0)

    Returns:
        Node: Line of best fit for X, Y values
    """

    X = X.reshape(len(X), 1)
    Y = Y.reshape(len(Y), 1)

    reg = linear_model.LinearRegression()
    reg.fit(X, Y)
    node = Node()

    node.m = reg.coef_[0][0]  # slope
    node.b = reg.intercept_[0]  # intercept
    pred = X * node.m + node.b  # gen prediction array using mx+b formula for Y values
    node.min = np.min(pred)
    node.max = np.max(pred)

    # if desired depth not reached
    if current_d != d and node.max - node.min > 0:
        bins = np.floor(((pred - node.min) / (node.max - node.min)) * (w-1))
        bins = bins.astype(int)
        for wi in range(w):
            mask = bins == wi  # list of bins which equal wi
            Xwi = X[mask]  # list of values that fall into the bin
            Ywi = Y[mask]

            if len(Xwi) > 0 and len(Ywi) > 0:
                child_node = build_recursive(Xwi, Ywi, w, d, current_d + 1)
                node.children.append(child_node)
            else:
                node.children.append(None)

    return node


def predict_recursive(x, w, d, node):
    pred = x * node.m + node.b

    if len(node.children) > 0 and node.max - node.min > 0:
        bin = np.floor(((pred - node.min) / (node.max - node.min)) * (w-1))
        bin = bin.astype(int)
        if bin >= 0 and \
                len(node.children) > bin and \
                node.children[bin] is not None:
            pred = predict_recursive(x, w, d, node.children[bin])

    return pred


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file", type="str",
                      default="data_1.csv", help="data file")
    parser.add_option("-d", "--depth", dest="depth", type="int",
                      default=2, help="depth of the model tree")
    parser.add_option("-w", "--width", dest="width", type="int",
                      default=10, help="width of the model layers")

    (options, args) = parser.parse_args()

    # load csv and columns
    df = pd.read_csv(options.file)
    Y = df['pos']
    X = df['value']
    X = X.values.reshape(len(X), 1)
    Y = Y.values.reshape(len(Y), 1)

    # setup figures
    f1 = plt.figure()
    f2 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax2 = f2.add_subplot(111)
    ax2.set_xlabel('age')
    ax2.set_ylabel('count')
    ax1.set_xlabel('age')
    ax1.set_ylabel('pos')

    # plot "true" cdf
    ax1.scatter(X, Y, color='g', alpha=0.5, s=4)

    # build learned index model
    d = options.depth  # depth of the recursion
    w = options.width  # width of the layers
    node = build_recursive(X, Y, w, d, 0)
    print("number of nodes in model = " + str(Node.node_count))

    # predict the cdf
    predictions = []
    testX = np.linspace(np.min(X), np.max(X), 10000)
    for i in range(len(testX)):
        pred = predict_recursive(testX[i], w, d, node)
        predictions.append(pred)

    # plot the predicted cdf
    ax1.plot(testX, predictions, color='blue', linewidth=1)

    # plot the histogram
    n, bins, patches = ax2.hist(X, 20, facecolor='g', alpha=0.5)

    # predict the histogram
    counts = []
    predicted_hist = []
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        pred_upper = predict_recursive(upper, w, d, node)
        pred_lower = predict_recursive(lower, w, d, node)
        c = pred_upper - pred_lower

        predicted_hist.append(lower)
        predicted_hist.append(upper)
        counts.append(c)
        counts.append(c)
        print("count of " + str(lower) + " to " + str(upper) + " = " + str(c))
    ax2.plot(predicted_hist, counts, color='blue', linewidth=1)

    plt.show()
