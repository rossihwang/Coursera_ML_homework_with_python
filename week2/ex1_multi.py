#!/usr/bin/python3
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
from func import featureNormalize, fmin, normalEqn

def main():
    ## Part 1: Feature Normalization
    print("Loading data ...")

    # Load data
    data = np.loadtxt("./ex1data2.txt", delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    m = y.size

    # Print out some data points
    print("First 10 examples from the dataset: ")
    #TODO How to implement a print with format for array like that in Matlab?
    for i, j in zip(X[:10, :], y.reshape(-1, 1)[:10, :]):
        print(" x = [{:.0f} {:.0f}], y = {:.0f}".format(i[0], i[1], j[0]))

    # print("Program paused. Press enter to continue. ")
    # input() 

    # Scale features and set them to zero mean
    print("Normalizing Feature ...")

    X, mu, sigma = featureNormalize(X)

    # Add intercept term to X
    X = np.hstack([np.ones((m, 1)), X])

    ## Part 2: Gradient Descent
    print("Running gradient descent ...")
    
    # Choose some alpha value
    alpha = 0.1
    num_iters = 50

    # Init Theta and Run Gradient Descent
    theta = np.zeros((3, 1))
    # Reuse gradientDescent 
    theta, J_history = fmin(X, y, theta, alpha, num_iters)

    # Plot the convergence graph
    plt.figure()
    print(J_history)
    plt.plot(J_history, "-b", linewidth=2)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost J")

    # Display gradient descent's result 
    print("Theta computed from gradient descent: ")
    print(theta)
    print() 

    # Estimate the price of a 1650 sq-ft, 3 br house
    price = np.hstack([1, (1650-mu[0])/sigma[0], (3-mu[1])/sigma[1]]) @ theta

    print("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n{}".format(price))
    # print("Program paused. Press enter to continue.")
    # input()

    ## Part 3: Normal Equations
    # Load Data
    data = np.loadtxt("./ex1data2.txt", delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    m = y.size 

    # Add intercept term to X 
    X = np.hstack([np.ones((m, 1)), X])

    # Calculate the parameters from the normal equation
    theta = normalEqn(X, y)

    print("Theta computed from the normal equations:\n{}".format(theta))
    print()

    # Estimate the price of a 1650 sq-ft, 3 br house 
    price = np.array([1, 1650, 3]) @ theta 

    print("Predicted price of a 1650 sq-ft, 3 br house (using normal equations:\n {})".format(price))
    plt.show()

if __name__ == "__main__":
    main()