#!/usr/bin/python3
import numpy as np 
import matplotlib.pyplot as plt 
from func import plotData, mapFeature, costFunctionReg, gradientReg, scipy_fminunc, plotDecisionBoundary, predict
import scipy.optimize as opt


def main():
    ## Load Data
    # The first two columns contains the exam scores and the third column 
    # contains the label. 

    data = np.loadtxt("./ex2data2.txt", delimiter=',')
    X = data[:, :2]
    y = data[:, 2]

    plotData(X, y)
    # Labels and Legend
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.legend(["y = 1", "y = 0"])

    ## Part 1: Regularized Logistic Regression
    X = mapFeature(X[:, 0, np.newaxis], X[:, 1, np.newaxis])  # (100, 22)

    # Initialize fitting parameters
    initial_theta = np.zeros((X.shape[1], 1))

    # Set regularization parameter lambda to 1
    lmbd = 1

    # Compute and display initial cost and gradient for regularized logistic regression
    cost = costFunctionReg(initial_theta, X, y, lmbd)
    grad = gradientReg(initial_theta, X, y, lmbd)

    print("Cost at initial theta (zeros): {}".format(cost))
    print("Expected cost (approx): 0.693")
    print("Gradient at initial theta (zeros) - first five values only:")
    print(grad[:5])
    print("Expected gradients (approx) - first five values only:")
    print("0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115")

    # Compute and display cost and gradient
    # with all-ones theta and lambda = 10
    test_theta = np.ones((X.shape[1], 1))
    cost = costFunctionReg(test_theta, X, y, 10)
    grad = gradientReg(test_theta, X, y, 10)

    print("\nCost at test theta (with lambda = 10): {}".format(cost))
    print("Expected cost (approx): 3.16")
    print("Gradient at test theta -first five values only:")
    print(grad[:5])
    print("Expected gradients (approx) - first five values only:")
    print("0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922")

    ## Part 2: Regularization and Accuracies
    initial_theta = np.zeros((X.shape[1], 1))

    # Set regularization parameter lambda to 1
    lmbd = 1
    options = {"maxiter": 400, "disp": True}
    theta, cost = scipy_fminunc(costFunctionReg, initial_theta, (X, y, lmbd), options)

    # Plot Boundary
    plotDecisionBoundary(theta, X, y)
    plt.title("lambda = {}".format(lmbd))
    plt.xlabel("Microchip Test 1")
    plt.ylabel("Microchip Test 2")
    plt.legend(["y = 1", "y = 0", "Decision boudary"])

    # Compute accuracy on our training set
    p = predict(theta, X)

    print("Train Accuracy: {}".format(np.mean(np.float64(p == y.reshape(-1, 1))) * 100))
    print("Expected accuracy (with lambda = 1): 83.1 (approx)")

    plt.show()

if __name__ == "__main__":
    main()