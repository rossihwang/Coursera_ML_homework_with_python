#!/usr/bin/python3
import numpy as np 
from scipy.io import loadmat 
from func import displayData, predict
import matplotlib.pyplot as plt 

def main():

    input_layer_size = 400 # 20x20 Input Images of Digits
    hidden_layer_size = 25 # 25 hidden units
    num_labels = 10 # 10 labels, from 1 to 10

    ## Part 1: Loading and Visualizing Data
    print("Load and Visualizing Data ...")
    dat = loadmat("./ex3data1.mat")
    X = dat["X"]
    y = dat["y"]  # 5000x1
    m = X.shape[0]

    # Randomly select 100 data points to display
    rand_indices = np.random.permutation(m)
    sel = X[rand_indices[:100], :]

    displayData(sel)
    plt.show(block=True)

    ## Part 2: Loading Pameters
    print("Loading Saved Neural Network Parameters ...")
    dat1 = loadmat("./ex3weights.mat")
    Theta1 = dat1["Theta1"]
    Theta2 = dat1["Theta2"]

    ## Part 3: Implement Predic
    pred = predict(Theta1, Theta2, X)
    print("\nTraining Set Accuracy: {}".format(np.mean(np.float64(pred==y)) * 100))

    rp = np.random.permutation(m)
    for i in range(m):
        print("\nDisplaying Example Image")
        displayData(X[rp[i], np.newaxis, :])
        
        pred = predict(Theta1, Theta2, X[rp[i], np.newaxis, :])
        print("\nNeutral Network Prediction: {} (digit {})".format(pred.ravel(), pred.ravel() % 10))

        plt.show()

        # Pause with quit option
        s = input("Paused - press enter to continue, q to exit:")
        if s == "q":
            break

if __name__ == "__main__":
    main()