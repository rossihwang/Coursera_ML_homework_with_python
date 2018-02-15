import numpy as np 
from scipy.io import loadmat 
from func import plotData, svmTrain
import matplotlib.pyplot as plt 

def main():
    ## Part 1: Loading and Visualizing Data
    print("Loading and Visualizing Data ...")

    dat = loadmat("./ex6data1.mat")
    X = dat['X']
    y = dat['y']

    print(X.shape, y.shape)
    plotData(X, y)

    ## Part 2: Training Linear SVM 
    # dat = loadmat("./ex6data1.mat")
    print("\nTraining Linear SVM ...")

    C = 1000
    model = svm

    ## Part 3: Implementing Gaussian Kernel 


    ## Part 4: Visualizing Dataset 2

    ## Part 5: Training SVM with RBF Kernel (Dataset 2)

    ## Part 6: Visualizing Dataset 3

    ## Part 7: Training SVM with RBF Kernel (Dataset 3)

    plt.show()

if __name__ == "__main__":
    main()