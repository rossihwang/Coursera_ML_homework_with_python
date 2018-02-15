import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

def linearRegCostFunction(X, y, theta, lmbd):
    m = y.size

    grad = np.zeros_like(theta)
    h = X @ theta
    J = np.mean(np.square(h - y)) / 2 + (np.mean(np.square(theta[1:])) * lmbd) / 2
    grad = X.T @ (h - y) / m
    grad[1:] = grad[1:] + theta[1:] * lmbd / m
    return J, grad 

def trainLinearReg(X, y, lmbd):
    y = y.reshape(-1, 1)
    Xc = tf.constant(X, dtype=tf.float64, name="X")
    yc = tf.constant(y, dtype=tf.float64, name="y")
    lmbdc = tf.constant(lmbd, dtype=tf.float64, name="lambda")

    theta_init = np.zeros((X.shape[1], 1))
    theta = tf.Variable(theta_init, name="theta")

    pred = tf.matmul(X, theta)
    mse = tf.reduce_mean(tf.square(pred - yc) / 2, name="MSE")

    # Regularization
    if lmbd != 0.0:
        mse += lmbdc * tf.reduce_mean(tf.square(tf.slice(theta, [1, 0], [-1, 1]))) / 2

    training_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(mse)

    init = tf.global_variables_initializer()
    n_epochs = 400

    with tf.Session() as sess:
        sess.run(init)
        for ep in range(n_epochs):
            sess.run(training_op)
            if ep % 50 == 0:
                print("Epoch: {}, MSE: {}".format(ep, mse.eval()))

        theta_best = theta.eval() 

    return theta_best

def learningCurve(X, y, Xval, yval, lmbd):
    m = X.shape[0]
    error_train = np.zeros(m) 
    error_val = np.zeros(m)

    for i in range(m):
        theta = trainLinearReg(X[:i+1], y[:i+1], lmbd)
        error_train[i], _ = linearRegCostFunction(X[:i+1], y[:i+1], theta, 0)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta, 0)

    return error_train, error_val

def polyFeatures(X, p):
    X_poly = np.zeros((X.shape[0], p))

    for i in range(p):
        X_poly[:, i, np.newaxis] = np.power(X, i+1)    

    return X_poly

def plotFit(min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x-15, max_x+25+0.05, 0.05).reshape(-1, 1)

    X_poly = polyFeatures(x, p)
    X_poly = (X_poly - mu) / sigma 

    X_poly = np.hstack([np.ones((x.shape[0], 1)), X_poly])
    plt.plot(x, X_poly @ theta, "--", linewidth=2.0)

def validationCurve(X, y, Xval, yval):
    lmbd_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    error_train = np.zeros((lmbd_vec.size, 1))
    error_val = np.zeros((lmbd_vec.size, 1))

    for i in range(lmbd_vec.size):
        l = lmbd_vec[i]
        theta = trainLinearReg(X, y, l)
        error_train[i], _ = linearRegCostFunction(X, y, theta, 0)
        error_val[i], _ = linearRegCostFunction(Xval, yval, theta, 0)

    # for i in zip(lmbd_vec, error_train, error_val):
    #     l = i[0]
    #     theta = trainLinearReg(X, y, l)
    #     i[1], _ = linearRegCostFunction(X, y, theta, 0)
    #     i[2], _ = linearRegCostFunction(Xval, yval, theta, 0)

    return lmbd_vec, error_train, error_val