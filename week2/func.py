import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy.linalg as la 

def warmUpExercise():
    A = np.eye(5)
    print(A)

def plotData(X, y):
    plt.figure()
    plt.plot(X, y, "rx", markersize=10)
    
def computeCost(X, y, theta):
    # TODO reuse this function for numpy array and tensor?
    m = y.size 
    y = y.reshape(-1, 1)
    theta = theta.reshape(-1, 1)
    return np.sum(np.power((X @ theta - y), 2)) / (2*m)

def gradientDescent(X, y, theta, alpha, n_epochs):
    y = y.reshape(-1, 1)
    theta = theta.reshape(-1, 1)
    J_history = np.array([])
    m = y.size
    Xc = tf.constant(X, dtype=tf.float32, name="X")
    yc = tf.constant(y, dtype=tf.float32, name="y")
    theta = tf.Variable(theta, dtype=tf.float32, name="theta")
    y_pred = tf.matmul(Xc, theta, name="predictions")
    error = y_pred - yc
    mse = tf.reduce_mean(tf.square(error), name="mse") # divided by 2 in Matlab version
    # type 1
    # gradients = 1/m * tf.matmul(tf.transpose(Xc), error)
    # type 2
    # gradients = tf.gradients(mse, [theta])[0]
    # training_op = tf.assign(theta, theta - alpha * gradients)
    ### type 1 & 2 end ###
    # type 3
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    training_op = optimizer.minimize(mse)
    ### type 3 end ###

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