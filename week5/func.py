import numpy as np 
import matplotlib.pyplot as plt 

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
                         X[curr_ex, :].reshape(example_height, example_width) / (max_val + 1e-7)
            curr_ex += 1
        if curr_ex >= m:
            break 
    # plt.figure()
    plt.imshow(display_array.T, cmap="gray") # Why need transpose?
    plt.axis("off")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

def get_batch_index(data_size, batch_size):
    n_batch = data_size // batch_size
    res = data_size % batch_size

    for i in range(n_batch):
        yield np.arange(i*batch_size, (i+1)*batch_size)
    if res != 0:
        yield np.arange((i+1)*batch_size, data_size)

def fmin_nn(n_inputs, n_hidden, n_outputs, X_input, y_input, lmbd_input):
    """ Construct layers with fully_connected()
    Substitute two sigmoids with relu and softmax for enhancement respectively. Another 
    reason is a bit difficult to implement multi labels sigmoid in tensorflow.
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(X_input, y_input):
        X_train = X_input[train_idx]
        y_train = y_input[train_idx]
        X_test = X_input[test_idx]
        y_test = y_input[test_idx]
    y_train = y_train.ravel() % 10
    y_test = y_test.ravel() % 10
    y_input = y_input.ravel() % 10

    # X, y are not identify to the inputs
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')
    lmbd = tf.constant(lmbd_input, dtype=tf.float32, name="lambda")

    with tf.name_scope("dnn"):
        hidden = fully_connected(X, n_hidden, scope="hidden")
        outputs = fully_connected(hidden, n_outputs, scope="outputs", activation_fn=None)

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=outputs)
        loss = tf.reduce_mean(xentropy, name="loss")

    if lmbd_input != 0.0:
        with tf.variable_scope("hidden", reuse=tf.AUTO_REUSE):
            w1 = tf.get_variable("weights")
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w1)
        with tf.variable_scope("outputs", reuse=tf.AUTO_REUSE):
            w2 = tf.get_variable("weights")
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w2)
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        print(reg_var)
        print("hello")
        loss += (lmbd * tf.reduce_sum([tf.reduce_sum(tf.square(v)) for v in reg_var]))

    learning_rate = 0.01
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(outputs, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 200
    batch_size = 50
    Theta1_weights = np.zeros((n_hidden, (n_inputs+1)))
    Theta2_weights = np.zeros((n_outputs, (n_hidden+1)))
    Theta1_bias = np.zeros(n_hidden)
    Theta2_bias = np.zeros(n_outputs)

    with tf.Session() as sess:
        # saver.restore(sess, "./nn_model.ckpt")
        sess.run(init)
        for epoch in range(n_epochs):
            for idx in get_batch_index(X_train.shape[0], batch_size):
                X_batch, y_batch = X_train[idx], y_train[idx]
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if epoch % 50 == 0:
                acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
                print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        # for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        #     print(v)
        with tf.variable_scope("hidden", reuse=tf.AUTO_REUSE):
            Theta1_weights = tf.get_variable("weights").eval().T
            Theta1_bias = tf.get_variable("biases").eval()
        with tf.variable_scope("outputs", reuse=tf.AUTO_REUSE):
            Theta2_weights = tf.get_variable("weights").eval().T
            Theta2_bias = tf.get_variable("biases").eval()

    # Add biases
    Theta1 = np.hstack([Theta1_bias.reshape(-1, 1), Theta1_weights])
    Theta2 = np.hstack([Theta2_bias.reshape(-1, 1), Theta2_weights])
    nn_params = np.vstack([Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)]) 

    cost, _ = nnCostFunction(nn_params, n_inputs, n_hidden, n_outputs, X_input, y_input, lmbd_input)

    return nn_params, cost

from sklearn.preprocessing import OneHotEncoder
def fmin_nn1(n_inputs, n_hidden, n_outputs, X_input, y_input, lmbd_input):
    """Manual constructed layers
    """
    y_copy = y_input.copy()
    y_copy[y_input==10] = 0
    enc = OneHotEncoder(n_values=10, sparse=False)
    y_enc = enc.fit_transform(y_copy).reshape(-1, 10)
    
    X = tf.placeholder(tf.float32, [None, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_outputs])
    lmbd = tf.constant(lmbd_input, dtype=tf.float32, name="lambda")

    with tf.name_scope("layer1"):
        w1 = tf.Variable(tf.random_normal([n_inputs, n_hidden]), name="w1")
        b1 = tf.Variable(tf.random_normal([n_hidden]), name="b1")
        L1 = tf.nn.relu(tf.matmul(X, w1) + b1)

    with tf.name_scope("layer2"):
        w2 = tf.Variable(tf.random_normal([n_hidden, n_outputs]), name="w2")
        b2 = tf.Variable(tf.random_normal([n_outputs]), name="b2")
        hypothesis = tf.matmul(L1, w2) + b2

    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))
    
    if lmbd_input != 0.0:
        with tf.name_scope("regularization"):
            cost += lmbd * (tf.reduce_mean(tf.square(w1)) +tf.reduce_mean(tf.square(w2))) # L2

    with tf.name_scope("eval"):
        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    training_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    init = tf.global_variables_initializer()

    n_epochs = 200
    batch_size = 50

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for idx in get_batch_index(X_input.shape[0], batch_size):
                X_batch, y_batch = X_input[idx], y_enc[idx]
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if epoch % 50 == 0:
                accuracy_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                print(epoch, "Train accuracy:", accuracy_train)
        bias1 = b1.eval()
        weights1 = w1.eval()
        bias2 = b2.eval()
        weights2 = w2.eval()
    Theta1 = np.hstack([bias1.reshape(-1, 1), weights1.T])
    Theta2 = np.hstack([bias2.reshape(-1, 1), weights2.T])
    nn_params = np.vstack([Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)])

    # This cost is meaningless
    cost, _ = nnCostFunction(nn_params, n_inputs, n_hidden, n_outputs, X_input, y_input, lmbd_input)

    return nn_params, cost

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbd):
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = nn_params[:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size, (input_layer_size+1))
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels, (hidden_layer_size+1))
    # Theta1 = (25, 401)
    # Theta2 = (10, 26)
    # Setup some useful variables
    m = X.shape[0]

    J = 0
    Theta1_grad = np.zeros_like(Theta1)
    Theta2_grad = np.zeros_like(Theta2)

    ## Compute cost
    a_1 = np.hstack([np.ones((X.shape[0], 1)), X])
    z_2 = a_1 @ Theta1.T 
    a_2 = sigmoid(z_2)
    a_2 = np.hstack([np.ones((a_2.shape[0], 1)), a_2])
    z_3 = a_2 @ Theta2.T 
    a_3 = sigmoid(z_3) # 1x10

    y = y.ravel()
    # Compute loss
    for j in range(num_labels):
        J = J + np.sum(-np.log(a_3[y==(j+1), j])) + np.sum(-np.log(1-a_3[y!=(j+1), j]))
    J = (J + (lmbd / 2) * (np.sum(np.power(Theta1[:, 1:], 2)) + np.sum(np.power(Theta2[:, 1:], 2)))) / m

    ## Compute gradient
    for i, x in enumerate(X):
        ## Step1: Feedforward
        a_1 = np.append(np.ones(1), x) # 1x401
        z_2 = a_1 @ Theta1.T # 1x25
        a_2 = sigmoid(z_2)
        a_2 = np.append(np.ones(1), a_2) # 1x26
        z_3 = a_2 @ Theta2.T # 1x10
        a_3 = sigmoid(z_3)
        ## Step2: Compute cost in output layer
        yy = np.zeros_like(a_3)
        yy[y[i]-1] = 1
        delta_3 = a_3 - yy # 1x10
        ## Step3: Compute cost in hidden layer
        delta_2 = (delta_3 @ Theta2[:, 1:]) * sigmoidGradient(z_2) # 1x25
        ## Step4:
        Theta1_grad = Theta1_grad + delta_2.reshape(-1, 1) @ a_1.reshape(1, -1)
        Theta2_grad = Theta2_grad + delta_3.reshape(-1, 1) @ a_2.reshape(1, -1)
    ## Step5:
    Theta1_grad = Theta1_grad / m
    Theta2_grad = Theta2_grad / m

    ## Add regularization
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + Theta1[:, 1:] * (lmbd / m)
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + Theta2[:, 1:] * (lmbd / m)

    return J, np.vstack([Theta1_grad.reshape(-1, 1), Theta2_grad.reshape(-1, 1)])


def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z)) 

def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out, 1+L_in) * 2 * epsilon_init - epsilon_init

def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1+fan_in))
    return np.sin(np.arange(W.size)).reshape(W.shape) / 10

def checkNNGradients(lmbd=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X 
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + np.arange(1, m+1) % num_labels

    # Unroll parameters
    nn_params = np.vstack([Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)])

    # Short hand for cost function
    from functools import partial
    params = {"input_layer_size": input_layer_size, 
              "hidden_layer_size": hidden_layer_size, 
              "num_labels": num_labels, 
              "X": X, 
              "y": y, 
              "lmbd": lmbd}
    costFunc = partial(nnCostFunction, **params)
    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    print(numgrad, grad)
    print("""The above two columns you get should be very similar.
    (Left-Your Numerical Gradient, Right-Analytical Gradient)\n""")

    import numpy.linalg as la 
    diff = la.norm(numgrad-grad)/la.norm(numgrad+grad)

    print("""If your backpropagation implementation is correct, then
    the relative difference will be small (less than 1e-9).
    Relative Difference: {}""".format(diff))

def computeNumericalGradient(J, theta):
    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    e = 1e-4

    for i in range(theta.size):
        perturb[i] = e
        loss1, _ = J(theta - perturb)
        loss2, _ = J(theta + perturb)
        numgrad[i] = (loss2 - loss1) / (2*e)
        perturb[i] = 0
    return numgrad

def softmax(z):
    return np.exp(z) / (np.sum(np.exp(z)) + 1e-7)

def relu(z):
    return np.maximum(z, 0, z)

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    p = np.zeros((X.shape[0], 1))

    h1 = relu(np.hstack([np.ones((m, 1)), X]) @ Theta1.T)
    h2 = softmax(np.hstack([np.ones((m, 1)), h1]) @ Theta2.T)
    return np.argmax(h2, 1) 