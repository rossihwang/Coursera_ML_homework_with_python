#!/usr/bin/python3
import numpy as np 
from scipy.io import loadmat
from func import displayData, nnCostFunction, sigmoidGradient, randInitializeWeights,\
checkNNGradients, fmin_nn, fmin_nn1, predict
import matplotlib.pyplot as plt 

def main():
    

    # Setup the parameters you will use for this exercise
    input_layer_size = 400 # mnist dataset 20x20
    hidden_layer_size = 25
    num_labels = 10

    ## Part 1: Loading and Visualizing Data
    print("Loading and Visualizing Data ...")
    dat = loadmat("./ex4data1.mat")
    X = dat['X']
    y = dat['y']
    m = X.shape[0]

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]

    displayData(sel)

    ## Part 2: Loading Parameters
    # Load the weights into variables Theta1 and Theta2
    dat1 = loadmat("./ex4weights.mat")
    Theta1 = dat1["Theta1"]
    Theta2 = dat1["Theta2"]

    # Unroll parameters
    nn_params = np.vstack([Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)])

    ## Part 3: Compute Cost (Feedforward)
    print("\nFeedforward Using Neural Network ...")

    # Weight regularization parameter 
    lmbd = 0

    J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbd)

    print("Cost at parameters (loaded from ex4weights): {}\n\
          (this value should be about 0.2877629)".format(J))

    ## Part 4: Implement Regularization
    print("\nChecking Cost Function (w/ Regularization) ...")
    lmbd = 1

    J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbd)

    print("Cost at parameters (loaded from ex4weights): {}\n\
          (this value should be about 0.383770)".format(J))

    ## Part 5: Sigmoid Gradient
    print("\nEvaluationg sigmoid gradient...")

    g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
    print("Sigmoid gradient evaluated at [-1, -0.5, 0, 0.5, 1]:")
    print(g)
    print("\n")

    ## Part 6: Initializing Parameters
    print("\nInitializing Neural Network Parameters ...")

    # initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
    # initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

    # Unroll parameters
    # initial_nn_params = np.vstack([initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)])
    
    ## Part 7: Implement Backpropagation
    print("\nChecking Backpropagation...")

    checkNNGradients()

    ## Part 8: Implement Regularization
    print("\nChecking Backpropagation (w/ Regularization) ...")

    # Check gradients by running checkNNGradients
    lmbd = 3
    checkNNGradients(lmbd)

    # Also output the costFunction debugging values
    debug_J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbd)

    print("\n\nCost at (fixed) debugging parameters (w/ lambda = {}): {}"\
    "\n(for lambda = 3, this value should be about 0.576051)\n".format(lmbd, debug_J))

    ## Part 8: Training NN
    print("\nTraining Neural Network...")

    lmbd = 1 # TODO optimize() can't not work with regularization now, should be 1 here
    nn_params, _ = fmin_nn1(input_layer_size, hidden_layer_size, num_labels, X, y, lmbd)
    Theta1 = nn_params[:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size, (input_layer_size+1))
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels, (hidden_layer_size+1))

    ## Part 9: Visualize Weights
    print("\nVisualizing Neural Network ...")

    displayData(Theta1[:, 1:])

    ## Part 10: Implement Predict
    pred = predict(Theta1, Theta2, X)
    pred[pred==0] = 10 # label 10 is set to 0 in the nn model

    print("\nTraining Set Accuracy: {}".format(np.mean(np.double(pred == y.ravel())) * 100))

    plt.show()

if __name__ == "__main__":
    main()
