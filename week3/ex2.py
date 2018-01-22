#!/usr/bin/python3
import numpy as np 
import matplotlib.pyplot as plt 
from func import plotData, costFunction, scipy_fminunc, plotDecisionBoundary, gradient, sigmoid, predict
from func import tf_minimize
import scipy.optimize as opt


def main():
    ## Load Data
    # The first two columns contains the exam scores and the third column 
    # contains the label. 

    data = np.loadtxt("./ex2data1.txt", delimiter=',')
    X = data[:, :2]
    y = data[:, 2]

    ## Part 1: Plotting
    print("""Plotting data with + indicating (y = 1) examples and o 
            indicating (y = 0) examples.""")
    plotData(X, y)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend(["Admitted", "Not admitted"])

    ## Part 2: Compute Cost and Gradient

    # Setup the data matrix appropriately, and add ones for the intercept term
    m, n = X.shape

    # Add intercept term to x and X_test
    X = np.hstack([np.ones((m, 1)), X])

    # Initialize fitting parameters
    initial_theta = np.zeros((n+1, 1))

    # Compute and display initial cost and gradient
    cost = costFunction(initial_theta, X, y)
    grad = gradient(initial_theta, X, y)

    print("\nCost at initial theta (zeros): {}".format(cost))
    print("Expected cost (approx): 0.693")
    print("Gradient at initial theta (zeros):\n{}".format(grad))
    print("Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628")

    # Compute and display cost and gradient with non-zero theta
    test_theta = np.array([-24, 0.2, 0.2])
    cost = costFunction(test_theta, X, y)
    grad = gradient(test_theta, X, y)

    print("\nCost at test theta: {}".format(cost))
    print("Expected cost (approx): 0.218")
    print("Gradient at test theta:\n{}".format(grad))
    print("Expected gradients (approx):\n 0.043\n 2.566\n 2.647")

    ## Part 3: Optimizing using fminunc
    options = {"maxiter": 400}
    ## Two implementation here
    # theta, cost = scipy_fminunc(costFunction, initial_theta, (X, y), options)
    theta, cost = tf_minimize(X, y, initial_theta)

    # Print theta to screen
    print("Cost at theta found by fminunc: {}".format(cost))
    print("Expected cost (approx): 0.203")
    print("theta: {}".format(theta))
    print("Expected theta (approx): \n-25.161\n 0.206\n 0.201")

    # Plot Boundary
    plotDecisionBoundary(theta, X, y)

    # Put some labels
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")

    # Legend, specific for the exercise
    plt.legend(["Admitted", "Not admitted", "Decision Boundary"])
    plt.axis([30, 100, 30, 100])

    ## Part 4: Predict and Accuracies
    prob = sigmoid(np.array([1, 45, 85]) @ theta)
    print("For a student with scores 45 and 85, we predict an admission probability of {}".format(prob))
    print("Expected value: 0.775 +/- 0.002")

    # Compute accuracy on our training set
    p = predict(theta, X)

    print("Train Accuracy: {}".format(np.mean(np.float64(p == y.reshape(-1, 1))) * 100))
    print("Expected accuracy (approx): 89.0")

    plt.show()


if __name__ == "__main__":
    main()

