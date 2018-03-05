from scipy.io import loadmat
from func import findClosestCentroids, computeCentroids, runkMeans, kMeansInitCentroids
import numpy as np 
import cv2 
import matplotlib.pyplot as plt


def main():

    ## Part 1: Find Closest Centroids
    print("Finding closest centroids.\n")
    dat = loadmat("ex7data2.mat")
    X = dat["X"]
    # print(X)

    # Select an initial set of centroids
    K = 3
    initial_centroids = np.array([3, 3, 6, 2, 8, 5]).reshape(3, 2)
    idx = findClosestCentroids(X, initial_centroids)

    print("Closest centroids for the first 3 examples: ")
    print(idx[:3]+1)
    print("\n(the closest centroids should be 1, 3, 2 respectively)")

    ## Part 2: Compute Means 
    print("\nComputing centroids means.\n")
    centroids = computeCentroids(X, idx, K)

    print("Centroids computed after initial finding of closest centroids: ")
    print(centroids)
    print("\n(the centroids should be)")
    print(" [2.428301, 3.157924]")
    print(" [5.813503, 2.633656]")
    print(" [7.119387, 3.616684]\n")

    ## Part 3: K-Means Clustering
    print("\nRuning K-Means clustering on example dataset.\n")

    dat = loadmat("ex7data2.mat") 
    X = dat["X"]

    K = 3
    max_iters = 10 

    initial_centroids = np.array([3, 3, 6, 2, 8, 5]).reshape(3, 2)

    centroids, idx = runkMeans(X, initial_centroids, max_iters, True)
    print("\nK-Means Done.")

    ## Part 4: K-Means Clustering on Pixels
    print("\nRunning K-Means clustering on pixels from an image.\n")

    A = cv2.imread("bird_small.png")

    A = A / 255

    img_size = A.shape 

    X = A.reshape(img_size[0]*img_size[1], 3)

    K = 16
    max_iters = 10

    initial_centroids = kMeansInitCentroids(X, K)

    centroids, idx = runkMeans(X, initial_centroids, max_iters)

    ## Part 5: Image Compression 

    print("\nApplying K-Means to compress an image.\n")

    idx = findClosestCentroids(X, centroids)

    X_recovered = centroids[np.int32(idx), :]

    X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

    plt.subplot(121)
    plt.imshow(A[:, :, ::-1])
    plt.title("Original")

    plt.subplot(122)
    plt.imshow(X_recovered[:, :, ::-1])
    plt.title("Compressed, with {} colors.".format(K))

    plt.show()

if __name__ == "__main__":
    main()