#!/usr/bin/python3
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from func import linearRegCostFunction, trainLinearReg, learningCurve, \
    polyFeatures, plotFit, validationCurve
from sklearn.preprocessing import StandardScaler 

def main():
    ## Part 1: Loading and Visualizing Data
    # Load Training Data
    print("Loading and Visualizing Data ...\n")
    dat = loadmat("./ex5data1.mat")
    print(dat.keys())
    X = dat["X"]
    y = dat["y"]
    Xval = dat["Xval"]
    yval = dat["yval"]
    Xtest = dat["Xtest"]
    ytest = dat["ytest"]
    m = X.shape[0]

    # Plot training data
    plt.plot(X, y, "rx", markersize=10, linewidth=2.0)
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")

    ## Part 2: Regularized Linear Regression Cost
    theta = np.array([1, 1]).reshape(-1, 1)
    J, _ = linearRegCostFunction(np.hstack([np.ones((m, 1)), X]), y, theta, 1)
    print("""Cost at theta= [[1], [1]]: {}\n
    (this value should be about 303.993192)\n""".format(J))


    ## Part 3: Regularized Linear Regression Gradient
    theta = np.array([1, 1]).reshape(-1, 1)
    J, grad = linearRegCostFunction(np.hstack([np.ones((m, 1)), X]), y, theta, 1)
    print("""Gradient at theta = [[1], [1]]: [[{}], [{}]]\n
    (this value should be about [[-15.303016], [598.250744]])\n""".format(grad[0], grad[1]))

    ## Part 4: Train Linear Regression 
    lmbd = 0
    # print(X.shape, y.shape)
    # return 
    theta = trainLinearReg(np.hstack([np.ones((m, 1)), X]), y, lmbd)

    # Plot fit over the data
    plt.figure()
    plt.plot(X, y, "rx", markersize=10, linewidth=1.5)
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.plot(X, np.hstack([np.ones((m, 1)), X]) @ theta, "--", linewidth=2.0)

    ## Part 5: Learning Curve for Linear Regression
    lmbd = 0
    error_train, error_val = learningCurve(np.hstack([np.ones((m, 1)), X]), y,
                np.hstack([np.ones((Xval.shape[0], 1)), Xval]), yval, lmbd) 
    
    plt.figure()
    plt.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val)
    plt.title("Learning curve for linear regression")
    plt.legend(["Train", "Cross Validation"])
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.axis([0, 13, 0, 150])

    print("# Training Examples\tTrain Error\tCross Validation Error\n")
    for i in range(m):
        print(" \t{}\t\t{}\t{}\n".format(i, error_train[i], error_val[i]))
    
    ## Part 6: Feature Mapping for Polynomial Regression
    p = 8

    # Map X onto Polynomial Features and Normalize
    X_poly = polyFeatures(X, p)
    scaler = StandardScaler().fit(X_poly)
    X_poly = scaler.transform(X_poly)
    X_poly = np.hstack([np.ones((m, 1)), X_poly])

    X_poly_test = polyFeatures(Xtest, p)
    X_poly_test = scaler.transform(X_poly_test)
    X_poly_test = np.hstack([np.ones((X_poly_test.shape[0], 1)), X_poly_test])

    X_poly_val = polyFeatures(Xval, p)
    X_poly_val = scaler.transform(X_poly_val)
    X_poly_val = np.hstack([np.ones((X_poly_val.shape[0], 1)), X_poly_val])

    print("Normalized Training Example 1:")
    print("{} \n".format(X_poly[0]))

    ## Part 7: Learning Curve for Polynomial Regression 
    lmbd = 1
    theta = trainLinearReg(X_poly, y, lmbd)
    plt.figure()
    plt.plot(X, y, "rx", markersize=10, linewidth=1.5)
    plotFit(np.min(X, axis=0), np.max(X, axis=0), scaler.mean_, np.sqrt(scaler.var_), theta, p)
    plt.xlabel("Change in water level (x)")
    plt.ylabel("Water flowing out of the dam (y)")
    plt.title("Polynomial Regression Fit (lambda = {})".format(lmbd))

    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lmbd) 
    
    plt.figure()
    plt.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val)
    plt.title("Polynomial Regression Learning Curve (lambda = {})".format(lmbd))
    plt.legend(["Train", "Cross Validation"])
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.axis([0, 13, 0, 100])

    print("Polynomial Regression (lambda = {})\n".format(lmbd))
    print("# Training Examples\tTrain Error\tCross Validation Error")
    for i in range(m):
        print(" \t{}\t\t{}\t{}\n".format(i, error_train[i], error_val[i]))

    ## Part 8: Validation for Selecting Lambda 
    lmbd_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)
    plt.figure()
    plt.plot(lmbd_vec, error_train, lmbd_vec, error_val)
    plt.legend(["Train", "Cross Validation"])
    plt.xlabel("lambda")
    plt.ylabel("Error")

    print("lambda\t\tTraining Error\tValidation Error\n")
    for i in zip(lmbd_vec, error_train, error_val):
        print("{0[0]}\t{0[1]}\t{0[2]}".format(i))

    plt.show()

if __name__ == "__main__":
    main()

