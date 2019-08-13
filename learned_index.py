import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from random import randint
from optparse import OptionParser
from scipy.stats.stats import pearsonr   


class Node:
    """Represents a line.

    Children are finer grained Node(s).
    Max and Min are for ranges.
    m and b come from the equation y = mx + b
    """

    node_count = 0
    max_depth = 0
    sum_depth = 0

    def __init__(self):
        Node.node_count += 1
        self.m = 0
        self.b = 0
        self.children = []
        self.max = 0
        self.min = 0


def build_recursive(X, Y, w, d, current_d, eps, delta):
    """Recursively build Node datastructure.

    Args:
        X (np.Array): X values
        Y (np.Array): Y values   
        w (int): the number of buckets to create
        d (int): the number of levels of recursion
        current_d (int): current depth (usage is to call this with 0)
        eps (float): maximum error tolerance (e.g., 0.05) for the index
        delta (float): maximum probability that error > eps (e.g., 0.1)

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

    thresh = eps*len(X)
    numout = sum(i > thresh for i in pred-Y)/float(len(X))
   

    # if desired depth not reached
    if (current_d != d or numout >= delta)  and node.max - node.min > 0:
        bins = np.floor(((pred - node.min) / (node.max - node.min)) * (w-1))
        bins = bins.astype(int)
        for wi in range(w):
            mask = bins == wi  # list of bins which equal wi
            Xwi = X[mask]  # list of values that fall into the bin
            Ywi = Y[mask]

            if len(Xwi) > 0 and len(Ywi) > 0:
                child_node = build_recursive(Xwi, Ywi, w, d, current_d + 1, eps, delta)
                node.children.append(child_node)
            else:
                node.children.append(None)
    
    if Node.max_depth <= current_d:
        Node.max_depth = current_d +  1         
    
    Node.sum_depth += 2
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
    parser.add_option("-e", "--epsilon", dest="eps", type="float",
                      default=0.05, help="error tolerance")
    (options, args) = parser.parse_args()

    # load csv and columns
    df = pd.read_csv(options.file)
    Y = df['pos']
    X = df['value']
    size = len(Y)
    for i in range(0, 10):
        for k in range(0, 100):
            X = np.append(X, i + randint(0,10))
            Y = np.append(Y, size+ i*100+k)

    X = np.array(X)
    X = np.sort(X)
    Y = np.array(Y)
    print("Dataset has " + str(len(X)) + " elements.") 



    # setup figures
    f1 = plt.figure()
    # f2 = plt.figure()
    ax1 = f1.add_subplot(111)
    # ax2 = f2.add_subplot(111)
    # ax2.set_xlabel('age')
    # ax2.set_ylabel('count')
    ax1.set_xlabel('age')
    ax1.set_ylabel('pos')

    # plot "true" cdf
    ax1.scatter(X, Y, color='g', alpha=0.5, s=4)

    # build learned index model
    d = options.depth  # depth of the recursion
    w = options.width  # width of the layers
    eps = options.eps
    node = build_recursive(X, Y, w, d, 0, eps, 0.1)
    print("number of nodes in model = " + str(Node.node_count))
    print("max model depth = " + str(Node.max_depth))
    print("avg model depth = " + str(Node.sum_depth / float(Node.node_count)))

    # predict the cdf
    predictions = []
    testX = np.linspace(np.min(X), np.max(X), 10000)
    for i in range(len(X)):
        pred = predict_recursive(X[i], w, d, node)
        predictions.append(pred)

    # plot the predicted cdf
    ax1.plot(X, predictions, color='blue', linewidth=1)

    # # plot the histogram
    # n, bins, patches = ax2.hist(X, 20, facecolor='g', alpha=0.5)

    count = 0
    for i in range(len(X)):
        pred = int(round(np.round(predict_recursive(X[i], w, d, node))))
        for k in range(0, 1):
            if pred + k >= 0 and pred +k< len(X) and X[i] == X[pred+k]:
                count += 1
                break

    print("accuracy = " + str(count/float(len(X))))

    # predict the histogram
    # counts = []
    # predicted_hist = []
    # for i in range(len(bins) - 1):
    #     lower = bins[i]
    #     upper = bins[i + 1]
    #     pred_upper = predict_recursive(upper, w, d, node)
    #     pred_lower = predict_recursive(lower, w, d, node)
    #     c = pred_upper - pred_lower

    #     predicted_hist.append(lower)
    #     predicted_hist.append(upper)
    #     counts.append(c)
    #     counts.append(c)
    #     print("count of " + str(lower) + " to " + str(upper) + " = " + str(c))
    # ax2.plot(predicted_hist, counts, color='blue', linewidth=1)

    plt.show()
