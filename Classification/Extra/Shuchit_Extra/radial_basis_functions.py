import numpy as np
import pandas as pd
import math
import argparse
from numpy.linalg import norm, pinv


no_of_centers=20
input_features=2
output_features=1
centers = [np.random.uniform(0, 1, input_features) for i in range(no_of_centers)]
weights = np.random.random((no_of_centers, output_features))

def basis_function(x, mu, sigma):
    global input_features, output_features, centers, weigths
    return np.exp(-(norm(x - mu)**2)/(2*sigma))

def calculate_activation(X):
    global input_features, output_features, centers, weigths
    G = np.zeros((X.shape[0], no_of_centers), float)
    for ci, c in enumerate(centers):
        for xi, x in enumerate(X):
            G[xi, ci] = basis_function(c, x, 0.125)
    return G

def train(X, Y):
    global input_features, output_features, centers, weigths
    random_index = np.random.permutation(X.shape[0])[:no_of_centers]
    centers = [X[i,:] for i in random_index]

    print("center", centers)
    G = calculate_activation(X)
    # print(G)

    output_weights = np.dot(pinv(G), Y)

def test(X):
    global input_features, output_features, centers, weigths

    G = calculate_activation(X)
    Y = np.dot(G, weights)
    return Y

def main():
    global input_features,output_features,centers, weigths
    train_data = pd.read_csv("her.tra",delimiter="\s+",header=None)
    input_data = np.matrix(train_data.values[:,:-1])
    input_features = train_data.shape[1] - 1
    output_features = 1
    output_data = np.matrix(train_data.values[:,input_features:])
    no_of_samples = train_data.shape[0]

    train(input_data, output_data)
    test_data = pd.read_csv("her.tes",delimiter="\s+",header=None)
    input_data = np.matrix(test_data.values[:,:-1])
    input_features = test_data.shape[1] - 1
    output_features = 1
    output_data = np.matrix(test_data.values[:,input_features:])
    no_of_samples = test_data.shape[0]
    z = test(input_data)
    print(z.shape, output_data.shape)
    for i in range(100):
        print(z[i], output_data[i])

if __name__ == "__main__":
    main()
