import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize as opt
import tensorflow as tf

def plotData(X, y):
    plt.figure()
    plt.plot(X[y==1, 0], X[y==1, 1], "k+", linewidth=2, markersize=7)
    plt.plot(X[y==0, 0], X[y==0, 1], "ko", markerfacecolor='y', markersize=7)

def sigmoid(z):
    return 1 / ((1 + np.exp(-z)) + 0) # Add 0.0001 to avoid divided by zero

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
    # print(1-h)
    J = np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h))
    return J

def gradient(theta, X, y):
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    m = y.size
    h = sigmoid(X @ theta)
    grad = (X.T @ (h - y)) / m
    return grad

def scipy_fminunc(func, x0, args, options={}):
    """
    Simulate fminunc with scipy(BFGS).
    """
    res = opt.minimize(func, x0, args, method="BFGS")
    # print(res)
    bestX = res.x
    cost = func(bestX, *args)
    return bestX.reshape(x0.shape), cost

def tf_gd(X, y, theta, alpha=0.001, n_epochs=100000):
    """
    Logistic regression in tensor flow, It seems Gradient Descent need smaller alpha and larger n_epochs.
    """
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)

    Xc = tf.constant(X, dtype=tf.float32, name="X")
    yc = tf.constant(y, dtype=tf.float32, name="y")
    theta = tf.Variable(theta, dtype=tf.float32, name="theta")
    z = tf.matmul(Xc, theta)
    # mse = tf.reduce_mean(tf.to_float((-yc * tf.log(h)) - ((1.0-yc) * tf.log(1.0-h))))
    mse = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yc, logits=z, name="mse"))
    training_op = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(mse)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 10000 == 0:
                print("Epoch", epoch, "MSE = ", mse.eval())
            sess.run(training_op)

        best_theta = theta.eval()
    cost = costFunction(best_theta, X, y)
    return best_theta.reshape(theta.shape), cost

def tf_gd_reg(X, y, theta, lmbd, alpha=0.03, n_epochs=800):
    """
    Logistic regression with regularization in tensorflow.
    """
    theta = theta.reshape(-1, 1)
    n_theta_row = theta.shape[0]
    y = y.reshape(-1, 1)

    Xc = tf.constant(X, dtype=tf.float32, name="X")
    yc = tf.constant(y, dtype=tf.float32, name="y")
    lmbdc = tf.constant(lmbd, dtype=tf.float32, name="lambda")
    theta = tf.Variable(theta, dtype=tf.float32, name="theta")
    z = tf.matmul(Xc, theta)
    # mse = tf.reduce_mean(tf.to_float((-yc * tf.log(h)) - ((1.0-yc) * tf.log(1.0-h))))
    mse = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yc, logits=z, name="mse"))
    ## Graph for regularization
    theta1, theta2 = tf.split(theta, [1, n_theta_row-1], 0)
    sqrt_theta = tf.matmul(tf.transpose(theta2), theta2)
    reg = lmbdc / 2 * tf.reduce_mean(sqrt_theta)
    loss = mse + reg 
    training_op = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 50 == 0:
                print("Epoch", epoch, "LOSS = ", loss.eval())
            sess.run(training_op)

        best_theta = theta.eval()
    cost = costFunction(best_theta, X, y)
    return best_theta.reshape(theta.shape), cost

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
        plt.contour(u, v, z, levels=[0])

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