#!/usr/bin/python3
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from func import warmUpExercise, plotData, computeCost, gradientDescent

def main():
    ## Part 1: Basic Function
    print("Running warmUpExercise ...")
    print("5x5 Identity Matrix: ")
    warmUpExercise() 

    print("Program paused. Press enter to continue.")
    # input() 

    ## Part 2: Plotting
    print("Plotting Data ...")
    data = np.loadtxt("./ex1data1.txt", delimiter=',')
    X = data[:, 0]
    y = data[:, 1]
    m = y.size
    
    plotData(X, y)
    print("Program paused. Press enter to continue.")
    # input()

    ## Part 3: Cost and Gradient descent
    X = np.hstack([np.ones((m, 1)), data[:, 0].reshape(-1, 1)])
    theta = np.zeros((2, 1))
    
    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01 

    print("\nTesting the cost function ...")
    # compute and display initial cost 
    J = computeCost(X, y, theta)
    print("With theta = [[0],[0]]\nCost computed = {}".format(J))
    print("Expected cost value (approx) 32.07")

    # further testing of the cost function
    J = computeCost(X, y, np.array([-1, 2]))
    print("\nWith theta = [[-1], [2]]\nCost computed = {}".format(J))
    print("Expected cost value (approx) 54.24")

    # print("Program paused. Press enter to continue.")
    # input()

    print("\nRunning Gradient Descent ...")
    # run gradient descent
    theta, _ = gradientDescent(X, y, theta, alpha, iterations)

    # print theta to screen
    print("Theta found by gradient descent:", end='')
    print("{}".format(theta))
    print("Expected theta values (approx)", end='')
    print(" -3.6303 1.1664\n")

    # Plot the linear fit
    plt.plot(X[:, 1], X @ theta, '-')
    plt.legend(["Training data", "Linear regression"])

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.array([1, 3.5]) @ theta 
    print("For population = 35,000, we predict a profit of {}".format(predict1*10000))
    predict2 = np.array([1, 7.0]) @ theta 
    print("For population = 70,000, we predict a profit of {}".format(predict2*10000))

    # print("Program paused. Press enter to continue.")
    # input()

    ## Part 4: Visualizing J(theta_0, theta_1)
    print("Visualizing J(theta_0, theta_1) ...")

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    theta0_mg, theta1_mg = np.meshgrid(theta0_vals, theta1_vals) # 100*100
    # Notice the shape of theta!
    J_ravel = np.array([computeCost(X, y, np.vstack([t0, t1])) for t0, t1 in zip(np.ravel(theta0_mg), np.ravel(theta1_mg))])
    J_vals = J_ravel.reshape(theta0_mg.shape) # 100*100  

    ax.plot_surface(theta0_mg, theta1_mg, J_vals)
    ax.set_xlabel("theta_0")
    ax.set_ylabel("theta_1")

    # Contour plot
    plt.figure()
    plt.contour(theta0_vals, theta1_vals, J_vals)
    plt.plot(theta[0], theta[1], "rx", markersize=10, linewidth=2)
    plt.show()

if __name__ == "__main__":
    main()