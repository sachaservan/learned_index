import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn import linear_model, neighbors, datasets
from optparse import OptionParser
from hilbertcurve.hilbertcurve import HilbertCurve
from learned_index import Node, build_recursive, predict_recursive

def get_dist(p1, p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def get_nn(p1, X):
    nn = [0,0]
    min_dist = 2**64
    for p2 in X:
        dist = get_dist(p1, p2)
        if min_dist > dist:
            nn = p2
            min_dist = dist
    return nn

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("-r", "--recursive-depth", dest="depth", type="int",
                      default=2, help="depth of the model tree")
    parser.add_option("-w", "--width", dest="width", type="int",
                      default=10, help="width of the model layers")
    parser.add_option("-d", "--dimention", dest="dimention", type="int",
                      default=2, help="dimentionality of each point in the dataset")
    parser.add_option("-e", "--epsilon", dest="eps", type="float",
                      default=0.05, help="error tolerance")
    parser.add_option("-s", "--shift", action="store_true", dest="shift", default=False,
                      help="shift the data")
    (options, args) = parser.parse_args()
    
    p = 100 # number of hypercubes along each dimention
    dim = options.dimention # number of dimentions
    shift = options.shift
    scale = 1000

    r = options.depth  # depth of the recursion
    w = options.width  # width of the layers
    eps = options.eps

    # load the desired dataset
    dataset = datasets.load_digits()
    data = dataset.data[:, :dim] # we only take the first dim features.
    print(data)

    # mult by const to 'fix point encode' data since need integers for hilbert
    data = [[int(p * scale) for p in element] for element in data]

    # X is training data, Y is test data
    X = data[:len(data)//2]
    Y = data[len(data)//2:]

    # instance of a hilbert curve
    hilbert_curve = HilbertCurve(p, dim)
    
    # learned index models for each transformation of the data
    models = []
    D = []

    for i in range(dim):
        dists = []
        shifted = []

        for point in X:
            coords = point.copy()
         
            # get the hilbert ditance for the point
            if shift and i != 0:
                coords = (np.array(coords) + i*scale)
                coords = coords.tolist()
            
            shifted.append(coords)

            hd = hilbert_curve.distance_from_coordinates(coords)
            dists = np.append(dists, hd)

        # sort points based on hilbert distance
        D.append([x for _,x in sorted(zip(dists, shifted))]) 

        # sort the distnaces and construct the model
        dists.sort()

        # build learned index model
        node = build_recursive(dists, np.array(range(len(X))), w, r, 0, eps, 0.01)
        models.append(node)
        
        #print("number of nodes in model = " + str(Node.node_count))
    
    print("max model depth = " + str(Node.max_depth))
    print("avg model depth = " + str(Node.sum_depth / float(Node.node_count)))
    
    hits = 0

    for i in range(len(Y)):

        preds = []
        for j in range(dim):
            coords = Y[i].copy()

            if shift and j != 0:
                coords = (np.array(coords) + j*scale)
                coords = coords.tolist()
           
            dist = hilbert_curve.distance_from_coordinates(coords)
            pred = predict_recursive(dist, w, r, models[j])
            index = int(pred)

            if index >= 0 and index < len(D[j]):
                err = max(int(eps*len(X)), 1)
                for k in range(-err, err):
                    if index + k >= 0 and index + k < len(D[j]):
                        qcoords = D[j][index + k].copy()

                        if shift and j != 0:
                            qcoords = (np.array(qcoords) - j*scale)                 
                            qcoords = qcoords.tolist()

                        preds.append(qcoords)

        # print(preds)
        if len(preds) > 0:
            pred = get_nn(Y[i], preds)
            actual = get_nn(Y[i], X)
            if (actual == pred):
                hits += 1
            # print(f'Actual NN = {actual}, predicted NN = {pred}')

    print(f'accuracy = {hits/len(Y)}; data shifted? {shift}')

    exit(0)

  
