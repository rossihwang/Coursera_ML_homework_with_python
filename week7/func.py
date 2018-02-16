import matplotlib.pyplot as plt 
import numpy as np
from sklearn.svm import SVC
import re 

def plotData(X, y):
    plt.figure()
    pos = (y==1).ravel()
    neg = (y==0).ravel() 
    plt.plot(X[pos, 0], X[pos, 1], "k+", linewidth=1, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], "ko", markerfacecolor='y', markersize=7)

def visualizeBoundaryLinear(X, y, model):
    w = model.coef_.ravel()
    b = model.intercept_
    xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    yp = -(w[0] * xp + b)/w[1]
    plotData(X, y)
    plt.plot(xp, yp, "-b")

def gaussianKernel(x1, x2, sigma):
    x1 = x1.ravel()
    x2 = x2.ravel() 

    sim = np.exp(-np.sum(np.square(x1-x2))/(2*sigma**2))
    
    return sim 

def visualizeBoundary(X, y, model):
    plotData(X, y)

    x1plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100).T
    x2plot = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100).T 
    X1, X2 = np.meshgrid(x1plot, x2plot)
    
    vals = np.zeros_like(X1)
    for i in range(X1.shape[1]):
        this_X = np.hstack([X1[:, i, np.newaxis], X2[:, i, np.newaxis]])
        vals[:, i] = model.predict(this_X)

    plt.contour(X1, X2, vals, levels=[0.5], colors='b')

def dataset3Params(X, y, Xval, yval):
    """
    Implement a grid search
    """
    C = 0.3
    sigma = 0.1 

    C_list = np.linspace(0.1, 1, 10)
    sigma_list = np.linspace(0.1, 1, 10)
    best_score = 0

    print("Grid search:")
    for c in C_list:
        for s in sigma_list:
            print("C: {}, sigma: {}".format(c, s))
            g = 1 / np.square(s)
            clf = SVC(C=c, gamma=g, kernel="rbf")
            clf.fit(X, y)
            score = clf.score(Xval, yval)
            print("Score: {}".format(score))
            if score > best_score:
                C = c
                sigma = s 
    return C, sigma

def processEmail(email_contents):
    vocabList = getVocabList()
    word_indices = []

    ## Preprocess Email
    print(email_contents)
    # Strip all HTML
    email_contents = re.sub("<[^<>]+>", " ", email_contents)

    # Handle Numbers
    email_contents = re.sub("[0-9]+", "number", email_contents)

    # Handle URLS
    email_contents = re.sub("(http|https)://[^\s]*", "httpaddr", email_contents)

    # Handle Email Addresses
    email_contents = re.sub("[^\s]+@[^\s]+", "emailaddr", email_contents)
    
    # Handle $ sign
    email_contents = re.sub("[$]+", "dollar", email_contents)
    print(email_contents)

    ## Tokenize Email
    print("\n==== Processed Email ====\n")
    l = 0

    str_list = re.split(["` @$/#.-:&*+=[]?!(){},''>_<;%\n\r"], email_contents.strip())

    for s in str_list:
        # Remove any non alphanumeric characters
        s = re.sub("[^a-zA-Z0-9]", '', s)

        # Stem the word
        
    return word_indices

def getVocabList():
    vocabList = []
    with open("./vocab.txt") as f:
        while True:
            try:
                num, vocab = f.readline().split()
            except ValueError:
                break
            vocabList.append(vocab)
    return vocabList
