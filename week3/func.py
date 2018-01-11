import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize as opt

def plotData(X, y):
    plt.figure()
    plt.plot(X[y==1, 0], X[y==1, 1], "k+", linewidth=2, markersize=7)
    plt.plot(X[y==0, 0], X[y==0, 1], "ko", markerfacecolor='y', markersize=7)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# def costFunction(theta, X, y):
#     theta = theta.reshape(-1, 1)
#     y = y.reshape(-1, 1)
#     m = y.size
#     h = sigmoid(X @ theta)
#     J = np.mean((-y * np.log(h)) - ((1 - y) * (np.log(1 - h))))
#     grad = (X.T @ (h - y)) / m
#     return J, grad

def costFunction(theta, X, y):
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    m = y.size
    h = sigmoid(X @ theta)
    J = np.mean((-y * np.log(h)) - ((1 - y) * (np.log(1 - h))))
    return J

def gradient(theta, X, y):
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    m = y.size
    h = sigmoid(X @ theta)
    grad = (X.T @ (h - y)) / m
    return grad

def scipy_fminunc(func, x0, args, options={}):
    res = opt.minimize(func, x0, args, method="BFGS")
    # print(res)
    bestX = res.x
    cost = func(bestX, *args)
    return bestX.reshape(x0.shape), cost

def plotDecisionBoundary(theta, X, y):
    # Plot Data
    plotData(X[:, 1:3], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1])-2, np.max(X[:, 1])+2])
        # Calculate the decision boundary line
        plot_y = (-1/theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
        for i in range(u.size):
            for j in range(v.size):
                z[i, j] = mapFeature(u[i], v[j]) @ theta 

        z = z.T 
        # Plot z = 0
        # Notice you need to specify the range [0, 0]?
        plt.contour(u, v, z, linewidth=2, levels=[0])

def mapFeature(X1, X2):
    """MAPFEATURE Feature mapping function to polynomial feature 
    MAPFEATURE(X1, X2) maps the two input features
    to quadratic features used in the regularization exercise.

    Returen a new feature array with more features, comprising of
    X1, X2, X1**2, X2**2, X1*X2, X1*X2**2, etc...

    Inputs X1, X2 must be the same szie.
    """
    degree = 6
    out = np.ones(X1.shape) # add intercept term
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack([out, (np.power(X1, (i-j)) * np.power(X2, j))])
    return out

def predict(theta, X):
    return np.round(sigmoid(X @ theta))

def costFunctionReg(theta, X, y, lmbd):
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    m = y.size
    h = sigmoid(X @ theta)
    J = np.mean((-y * np.log(h)) - ((1 - y) * np.log(1 - h))) \
        + (lmbd/(2*m)) * (theta[1:].T @ theta[1:]) # Regulation excludes theta0
    return J

def gradientReg(theta, X, y, lmbd):
    m = y.size
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    h = sigmoid(X @ theta).reshape(-1, 1)
    grad = np.zeros_like(theta)
    grad[0] = (X[:, 0, np.newaxis].T @ (h - y)) / m
    grad[1:] = ((X[:, 1:].T @ (h - y) / m) + ((lmbd / m) * theta[1:]))
    return grad.reshape(-1, 1)