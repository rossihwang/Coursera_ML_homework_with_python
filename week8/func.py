import numpy.linalg as la 
import numpy as np
import matplotlib.pyplot as plt 

def findClosestCentroids(X, centroids):
    K = centroids.shape[0]

    idx = np.zeros((X.shape[0], 1))

    m = X.shape[0]

    
    for i, x in enumerate(X):
        resMin = 99999 # initilize to a large value
        idxMin = 0
        for j, c in enumerate(centroids):
            resTmp = la.norm(x-c)
            if resMin >= resTmp:
                resMin = resTmp 
                idxMin = j
        idx[i] = idxMin
    return idx.ravel()

def computeCentroids(X, idx, K):
    m, n = X.shape 
    # print(m, n)
    centroids = np.zeros((K, n)) 

    for i in range(K):
        centroids[i] = np.mean(X[idx==i], axis=0)
    
    return centroids

def runkMeans(X, initial_centroids, max_iters, plot_progress=None):
    if plot_progress == None:
        isPlotProg = False 
    else:
        isPlotProg = True
    
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids 
    idx = np.zeros((m, 1))

    plt.figure()
    for i in range(max_iters):
        print("K-Means iteration {}/{}".format(i, max_iters))

        idx = findClosestCentroids(X, centroids)

        if isPlotProg:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            plt.show(block=False)
            input("Press enter to continue.")

        
        centroids = computeCentroids(X, idx, K)

    return centroids, idx 

def plotProgresskMeans(X, centroids, previous, idx, K, i):
    plotDataPoints(X, idx, K)
    plt.plot(centroids[:, 0], centroids[:, 1], "x", markeredgecolor='k', markersize=10, linewidth=3) 

    for j in range(centroids.shape[0]):
        drawLine(centroids[j, :], previous[j, :])

    plt.title("Iteration number {}".format(i))

def plotDataPoints(X, idx, K):
    plt.scatter(X[:, 0], X[:, 1], 15)

def drawLine(p1, p2, **kwarg):
    plt.plot(np.hstack([p1[0], p2[0]]), np.hstack([p1[1], p2[1]]), "--k", **kwarg)  

def kMeansInitCentroids(X, K):
    centroids = np.zeros((K, X.shape[1]))

    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K], :]

    return centroids

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma 
    return X_norm, mu, sigma

def pca(X):
    m, n = X.shape 
    sigma = (X.T @ X) / m 
    U, S, _ = la.svd(sigma)
    return U, S 

def recoverData(Z, U, K):
    X_rec = Z @ U[:, :K].T 
    return X_rec 

def projectData(X, U, K):
    return X @ U[:, :K]

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
    h = plt.imshow(display_array.T, cmap="gray") # Why need transpose?
    plt.axis("off")

    return h, display_array