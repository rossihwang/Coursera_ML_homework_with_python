import numpy as np 
from scipy.io import loadmat 
from func import plotData, visualizeBoundaryLinear, gaussianKernel,\
        visualizeBoundary, dataset3Params
import matplotlib.pyplot as plt 
from sklearn.svm import SVC 


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
    clf = SVC(C=C, kernel="linear")
    clf.fit(X, y)
    visualizeBoundaryLinear(X, y, clf)

    ## Part 3: Implementing Gaussian Kernel 
    print("\nEvaluating the Gaussian Kernel ...")

    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2

    sim = gaussianKernel(x1, x2, sigma)

    print("""Gaussian Kernel between x1 = [[1], [2], [1]], x2 = [[0], [4], [-1]], sigma = {} :\n\t{}\n(for sigma = 2, this value should be about 0.324652)""".format(sigma, sim))

    ## Part 4: Visualizing Dataset 2
    print("Loading and Visualizing Data ...")
    dat = loadmat("./ex6data2.mat")
    X = dat['X']
    y = dat['y']
    plotData(X, y)

    ## Part 5: Training SVM with RBF Kernel (Dataset 2)
    print("\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...")
    dat = loadmat("./ex6data2.mat")
    X = dat['X']
    y = dat['y']

    # SVM Parameters
    C = 1
    sigma = 0.1
    gamma = 1 / np.square(sigma) 

    clf = SVC(C=C, gamma=gamma, kernel="rbf")
    clf.fit(X, y)
    visualizeBoundary(X, y, clf)

    ## Part 6: Visualizing Dataset 3
    print("Loading and Visualizing Data ...")
    dat = loadmat("./ex6data3.mat")
    X = dat['X']
    y = dat['y']
    plotData(X, y)

    ## Part 7: Training SVM with RBF Kernel (Dataset 3)
    dat = loadmat("./ex6data3.mat")
    X = dat['X']
    y = dat['y']
    Xval = dat['Xval']
    yval = dat['yval']

    C, sigma = dataset3Params(X, y, Xval, yval)

    gamma = 1 / np.square(sigma)
    clf = SVC(C=C, gamma=gamma, kernel="rbf")
    clf.fit(X, y)
    visualizeBoundary(X, y, clf)

    plt.show()

if __name__ == "__main__":
    main()