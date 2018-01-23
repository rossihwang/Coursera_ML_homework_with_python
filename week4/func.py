import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf

def displayData(X, example_width=None):
    # Set example_width automatically
    if example_width == None:
        example_width = np.int(np.round(np.sqrt(X.shape[1])))
    
    # Compute rows, cols
    m, n = X.shape 
    example_height = np.int(n / example_width)

    # Compute number of items to display
    display_rows = np.int(np.floor(np.sqrt(m)))
    display_cols = np.int(np.ceil(m / display_rows))

    # Between images padding 
    pad = 1

    # Setup blank display
    display_array = np.zeros((np.int(pad + display_rows * (example_height + pad)), 
                            np.int(pad + display_cols * (example_width + pad))))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break
            # Copy the patch 
            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, :])) # Use for normalization
            row_idx = lambda x: (pad + j * (example_height + pad) + x) 
            col_idx = lambda x: (pad + i * (example_width + pad) + x)
            # print(row_idx(0))
            # print(row_idx(example_height))
            display_array[row_idx(0):row_idx(example_height),
                          col_idx(0):col_idx(example_width)] = \
                         X[curr_ex, :].reshape(example_height, example_width) / max_val
            curr_ex += 1
        if curr_ex >= m:
            break 
    # plt.figure()
    plt.imshow(display_array.T, cmap="gray") # Why need transpose?
    plt.axis("off")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def lrCostFunction(theta, X, y, lmbd):
    theta = theta.reshape(-1, 1)
    y = y.reshape(-1, 1)

    m = y.size
    grad = np.zeros_like(theta)
    
    h = sigmoid(X @ theta)
    J = np.mean((-y * np.log(h)) - ((1-y) * np.log(1-h))) + \
        (lmbd/(2*m) * (theta[1:].T @ theta[1:]))
    grad[0] = (X[:, 0, np.newaxis].T @ (h-y)) / m 
    grad[1:] = ((X[:, 1:].T @ (h-y)) + lmbd * theta[1:]) / m

    return J, grad

def tf_minimize_reg(X, y, theta, lmbd, alpha=0.03, n_epochs=3000):
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
    _, theta1 = tf.split(theta, [1, n_theta_row-1], 0)
    inner_prod_theta = tf.matmul(tf.transpose(theta1), theta1)
    reg = lmbdc / (2 * (n_theta_row-1)) * inner_prod_theta
    loss = mse + reg 

    # training_op = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss)
    training_op = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 1000 == 0:
                print("\bEpoch", epoch, "LOSS = ", loss.eval())
            sess.run(training_op)

        best_theta = theta.eval()
    return best_theta.flatten()

def oneVsAll(X, y, num_labels, lmbd):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n+1))
    X = np.hstack([np.ones((m, 1)), X])
    init_theta = np.zeros((n+1, 1))

    for c in np.arange(10):
        print("Class {}:".format(c))
        all_theta[c] = tf_minimize_reg(X, np.double(y==(c+1)), init_theta, lmbd)

    return all_theta

def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]

    p = np.zeros((X.shape[0], 1))
    X = np.hstack([np.ones((m, 1)), X])

    return (np.argmax(sigmoid(X @ all_theta.T), 1) + 1).reshape(-1, 1) # return the max index+1(the label)

def predict(theta1, theta2, X):
    m = X.shape[0]

    X = np.hstack([np.ones((m, 1)), X]) # Add bias
    a1 = sigmoid(X @ theta1.T)

    a1 = np.hstack([np.ones((m, 1)), a1])
    a2 = sigmoid(a1 @ theta2.T)

    return (np.argmax(a2, 1) + 1).reshape(-1, 1)
