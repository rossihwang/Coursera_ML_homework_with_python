import matplotlib.pyplot as plt 
from sklearn.svm import LinearSVC 

def plotData(X, y):
    plt.figure()
    pos = (y==1).ravel()
    neg = (y==0).ravel() 
    plt.plot(X[pos, 0], X[pos, 1], "k+", linewidth=1, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], "ko", markerfacecolor='y', markersize=7)

def svmTrain(X, y, C, kernel):
    pass 


def visualizeBoundaryLinear(X, y, model):
    pass