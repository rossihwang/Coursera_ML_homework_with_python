#!/usr/bin/python3
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
from func import displayData, lrCostFunction, oneVsAll, predictOneVsAll

def main():
    num_labels = 10

    ## Part 1: Loading and Visualizing Data
    print("Load and Visualizing Data ...")
    dat = loadmat("./ex3data1.mat")
    X = dat["X"]
    y = dat["y"]
    m = X.shape[0]

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]

    displayData(sel)

    ## Part 2a: Vectorize Logistic Regression
    print("Testing lrCostFunction() with regularization")

    theta_t = np.array([-2, -1, 1, 2], np.float).reshape(-1, 1)
    X_t = np.hstack([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3)/10])
    y_t = np.array([1, 0, 1, 0, 1], np.float).reshape(-1, 1)
    lmbd_t = 3
    J, grad = lrCostFunction(theta_t, X_t, y_t, lmbd_t)

    print("\nCost: {}".format(J))
    print("Expected cost: 2.534819")
    print("Gradients:")
    print(grad)
    print("Expected gradients:")
    print(" 0.146561\n -0.548558\n 0.724722\n 1.398003")

    ## Part 2b: One-vs-All Training 
    print("Training One-vs-All Logistic Regression...")
    lmbd = 0.1
    all_theta = oneVsAll(X, y, num_labels, lmbd)

    ## Part 3: Predict for One-Vs-All
    pred = predictOneVsAll(all_theta, X)

    print("Training Set Accuracy: {}".format(np.mean(np.float64(pred == y)) * 100)) # 94.94 in Matlab

    plt.show()

if __name__ == "__main__":
    main()