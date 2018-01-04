import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy.linalg as la 

def warmUpExercise():
    A = np.eye(5)
    print(A)

def plotData(X, y):
    plt.figure()
    plt.plot(X, y, "rx")
    
def computeCost(X, y, theta):
    # TODO reuse this function for numpy array and tensor?
    m = y.size 
    return np.sum(np.power((X @ theta - y), 2)) / (2*m)

def gradientDescent(X, y, theta, alpha, n_epochs):
    J_history = np.array([])
    m = y.size
    X = tf.constant(X, dtype=tf.float32, name="X")
    y = tf.constant(y, dtype=tf.float32, name="y")
    theta = tf.Variable(theta, dtype=tf.float32, name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse") # divided by 2 in Matlab version
    gradients = 1/m * tf.matmul(tf.transpose(X), error)
    training_op = tf.assign(theta, theta - alpha * gradients)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE = ", mse.eval())
            sess.run(training_op)
            J_history = np.append(J_history, mse.eval())
        best_theta = theta.eval()
        
    return best_theta.flatten(), J_history

def featureNormalize(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X - mu) / sigma 

    return X_norm, mu, sigma 

def normalEqn(X, y):
    return la.pinv(X.T @ X) @ X.T @ y