import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
from optparse import OptionParser


class Node:
    node_count = 0

    def __init__(self):
        Node.node_count += 1
        self.m = 0
        self.b = 0
        self.children = []
        self.max = 0
        self.min = 0


def chunk(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def build_model_2D(X, Y, P, w, d, g):
    # learn the X attribute
    node_x = build_recursive(X, P, w, d, 0)
    nodes_y = []

    prec = int(np.floor((g/100.0)*len(Y)))
    for yi in chunk(Y, prec):
        yi = yi[yi[:, 0].argsort()]
        node_yi = build_recursive(yi, np.array(range(0, prec)), w, d, 0)
        nodes_y.append(node_yi)

    return node_x, nodes_y


def build_recursive(V, P, w, d, current_d):
    V = V.reshape(len(V), 1)
    P = P.reshape(len(P), 1)

    # generate model for X pos
    reg = linear_model.LinearRegression()
    reg.fit(V, P)
    node = Node()

    # slope
    node.m = reg.coef_[0][0]

    # intercept
    node.b = reg.intercept_[0]

    # gen prediction array using mx+b formula for X values
    pred = V * node.m + node.b
    node.min = np.min(pred)
    node.max = np.max(pred)

    # if desired depth not reached
    if current_d != d and node.max - node.min > 0:
        bins = np.floor(((pred - node.min) / (node.max - node.min)) * (w-1))
        bins = bins.astype(int)
        for wi in range(w):
            mask = bins == wi  # list of bins which equal wi
            Vwi = V[mask]  # list of values that fall into the bin
            Pwi = P[mask]

            if len(Vwi) > 0 and len(Pwi) > 0:
                child_node = build_recursive(Vwi, Pwi, w, d, current_d + 1)
                node.children.append(child_node)
            else:
                node.children.append(None)

    return node


def predict_2D(x_min, x_max, y_min, y_max, w, d, node_x, nodes_y):
    pred_lower = predict_recursive(x_min, w, d, node_x)
    pred_upper = predict_recursive(x_max, w, d, node_x)

    print("prediction upper " + str(pred_upper))
    print("prediction lower " + str(pred_lower))

    prec = int(np.floor((g/100.0)*len(Y)))

    bin_lower = int(np.floor(pred_lower) / prec)
    bin_upper = int(np.floor(pred_upper) / prec)
    print("bin range: " + str(bin_upper - bin_lower))
    y_ranges = []

    for i in range(bin_lower, bin_upper):
        lower = predict_recursive(y_min, w, d, nodes_y[i])
        upper = predict_recursive(y_max, w, d, nodes_y[i])
        print("u: " + str(upper) + " l: " + str(lower))
        y_ranges.append((max(0, lower), max(0, min(upper, prec))))

    return pred_lower, pred_upper, y_ranges


def predict_recursive(v, w, d, node):
    pred = v * node.m + node.b

    if len(node.children) > 0 and node.max - node.min > 0:
        bin = np.floor(((pred - node.min) / (node.max - node.min)) * (w-1))
        bin = bin.astype(int)
        if bin >= 0 and len(node.children) > bin and node.children[bin] != None:
            pred = predict_recursive(v, w, d, node.children[bin])

    return pred


if __name__ == "__main__":

    x_min = 50
    x_max = 113
    y_min = 900000
    y_max = 1450000
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file", type="str",
                      default="data_2d.csv", help="data file")
    parser.add_option("-d", "--depth", dest="depth", type="int",
                      default=2, help="depth of the model tree")
    parser.add_option("-w", "--width", dest="width", type="int",
                      default=10, help="width of the model layers")
    parser.add_option("-g", "--gran", dest="gran", type="float",
                      default=1, help="granularity (in percent) of each bin")

    (options, args) = parser.parse_args()

    # load csv and columns
    df = pd.read_csv(options.file)
    P = df['pos']
    X = df['attr1']
    Y = df['attr2']
    X = X.values.reshape(len(X), 1)
    Y = Y.values.reshape(len(Y), 1)
    P = P.values.reshape(len(P), 1)

    # setup figures
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.set_xlabel('age')
    ax1.set_ylabel('count')

    # build learned index model
    d = options.depth  # depth of the recursion
    w = options.width  # width of the layers
    g = options.gran

    node_x, nodes_y = build_model_2D(X, Y, P, w, d, g)
    print("number of nodes in model = " + str(Node.node_count))

    mask = (X >= x_min) & (X <= x_max)
    Xm = X[mask]
    Ym = Y[mask]
    mask = (Ym >= y_min) & (Ym <= y_max)
    Xm = Xm[mask]
    Ym = Ym[mask]

    # plot the histogram
    n, bins, patches = ax1.hist(Xm, w, facecolor='g', alpha=0.5)

    # predict the histogram
    counts = []
    predicted_hist = []
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        pred_lower, pred_upper, y_ranges = predict_2D(
            lower, upper, y_min, y_max, w, d, node_x, nodes_y)

        c = 0
        for r in y_ranges:
            c += r[1] - r[0]

        predicted_hist.append(lower)
        predicted_hist.append(upper)
        counts.append(c)
        counts.append(c)
        print("count of " + str(lower) + " to " + str(upper) + " = " + str(c))

    ax1.plot(predicted_hist, counts, color='blue', linewidth=1)

    plt.show()
