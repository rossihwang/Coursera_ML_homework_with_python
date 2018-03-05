from scipy.io import loadmat 
import matplotlib.pyplot as plt 
from func import featureNormalize, drawLine, pca, recoverData, projectData, displayData
import numpy as np 

def main():
    ## Part 1: Load Example Dataset
    print("Visualizing example dataset for PCA.\n")

    dat = loadmat("ex7data1.mat")
    X = dat["X"]

    plt.plot(X[:, 0], X[:, 1], "bo")
    plt.axis([0.5, 6.5, 2, 8])

    ## Part 2: Pringcipal Component Analysis
    print("\nRunning PCA on example dataset.\n")

    X_norm, mu, sigma = featureNormalize(X)

    U, S = pca(X_norm)
    S = np.diag(S)
    # print(U, S)

    drawLine(mu, mu + 1.5 * S[0, 0] * U[:, 0])
    drawLine(mu, mu + 1.5 * S[1, 1] * U[:, 1])

    print("Top eigenvector: ")
    print(" U[:, 0] = {} {}".format(U[0, 0], U[1, 0]))
    print("\n(you should expect to see -0.707107 -0.707107)")


    ## Part 3: Dimension Reduction
    print("\nDimension reduction on example dataset.\n")
    plt.figure()
    plt.plot(X_norm[:, 0], X_norm[:, 1], "bo")
    plt.axis([-4, 3, -4, 3])
    plt.axis("square")

    K = 1
    Z = projectData(X_norm, U, K)
    print("Projection of the first example: {}".format(Z[0]))
    print("\n(this value should be about 1.481274)\n")

    X_rec = recoverData(Z, U, K)
    print("Approximation of the first example: {} {}".format(X_rec[0, 0], X_rec[0, 1]))
    print("\n(this value should be about -1.047419 -1.047419)\n")

    plt.plot(X_rec[:, 0], X_rec[:, 1], "ro")
    for i in range(X_norm.shape[0]):
        drawLine(X_norm[i, :], X_rec[i, :])

    ## Part 4: Loading and Visualizing Face Data
    print("Loading face dataset.\n")
    dat = loadmat("ex7faces.mat")
    X = dat["X"]

    plt.figure()
    displayData(X[:100, :])

    ## Part 5: PCA on Face Data: Eigenfaces
    print("\nRunning PCA on face dataset.\n(this might take a minute or two ...)")
    X_norm, mu, sigma = featureNormalize(X)

    U, S = pca(X_norm)
    plt.figure()
    displayData(U[:, :36].T)

    ## Part 6: Dimension Reduction for Faces
    print("\nDimension reduction for face dataset.\n")

    K = 100
    Z = projectData(X_norm, U, K)

    print("The projected data Z has a size of: ")
    print(Z.shape)

    ## Part 7: Visualization of Faces after PCA Dimension Reduction
    print("\nVisualizing the projected (reduce dimension) faces.\n")
    K = 100
    X_rec = recoverData(Z, U, K)

    plt.figure()
    plt.subplot(121)
    displayData(X_norm[:100, :])
    plt.title("Original faces")
    #plt.axis("square")

    plt.subplot(122)
    displayData(X_rec[:100, :])
    plt.title("Recovered faces")
    #plt.axis("square")

    ## Part 8: PCA for Visualization
    
    plt.show()

if __name__ == "__main__":
    main()