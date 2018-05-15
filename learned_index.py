import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
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

def build_recursive(X, Y, w, d, current_d):
    X = X.reshape(len(X),1)
    Y = Y.reshape(len(Y),1)
    
    reg = linear_model.LinearRegression()
    reg.fit(X, Y)
    node = Node()   
    
    node.m = reg.coef_[0][0]
    node.b = reg.intercept_[0]
    pred = X * node.m + node.b
    node.min = np.min(pred)
    node.max = np.max(pred)
    
    if current_d != d and node.max - node.min > 0:
        bins = np.floor(((pred - node.min) / (node.max - node.min)) * (w-1))
        bins = bins.astype(int)
        for wi in range(w):
            mask = bins == wi
            Xwi = X[mask]
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
        if bin >= 0 and len(node.children) > bin and node.children[bin] != None:
            pred = predict_recursive(x, w, d, node.children[bin])
        
        
    return pred
            
if __name__ == "__main__": 
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="file", type="str", default="data_1.csv", help="data file")
    parser.add_option("-d", "--depth", dest="depth", type="int", default=2, help="depth of the model tree")
    parser.add_option("-w", "--width", dest="width", type="int", default=10, help="width of the model layers")

    (options, args) = parser.parse_args()

    # load csv and columns
    df = pd.read_csv(options.file)
    Y = df['pos']
    X = df['value']
    X=X.values.reshape(len(X),1)
    Y=Y.values.reshape(len(Y),1)

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
    d = options.depth # depth of the recursion
    w = options.width # width of the layers
    node = build_recursive(X, Y, w, d, 0)
    print("number of nodes in model = " + str(Node.node_count))

    # predict the cdf
    predictions = []
    testX = np.linspace(np.min(X), np.max(X), 10000)
    for i in range(len(testX)):
        pred = predict_recursive(testX[i], w, d, node)
        predictions.append(pred)
     
    # plot the predicted cdf
    ax1.plot(testX, predictions, color='black',linewidth=1)

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
        print("count of " + str(lower) + " - " + str(upper) + " = " + str(c))
    ax2.plot(predicted_hist, counts, color='black', linewidth=1)

    plt.show()