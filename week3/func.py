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
    grad = (X.T @ (h - y)) / m
    return J

def gradient(theta, X, y):
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    m = y.size
    h = sigmoid(X @ theta)
    grad = (X.T @ (h - y)) / m
    return grad

def fminunc(func, x0, args):
    res = opt.minimize(func, x0, args, method="nelder-mead")
    # print(res)
    bestX = res.x
    cost = func(bestX, args[0], args[1])
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
        for i in u.size:
            for j in v.size:
                z[i, j] = mapFeature(u[i], v[j]) * theta 

        z = z.T 
        # Plot z = 0
        # Notice you need to specify the range [0, 0]?
        plt.contour(u, v, z, linewidth=2)

def mapFeature(X1, X2):
    degree = 6
    out = np.array([])
    for i in range(degree):
        for j in i:
            out = np.hstack([out, (np.power(X1, (i-j)) * np.power(X2, j))]) if out.size else (np.power(X1, (i-j)) * np.power(X2, j))
    return out

def predict(theta, X):
    return np.round(sigmoid(X @ theta))