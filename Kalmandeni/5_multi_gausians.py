import numpy as np
import matplotlib.pyplot as plt
def plot_correlated_data(X, Y, xlabel=None,
                         ylabel=None, equal=True):

    """Plot correlation between x and y by performing
    linear regression between X and Y.
    X: x data
    Y: y data
    xlabel: str
        optional label for x axis
    ylabel: str
        optional label for y axis
    equal: bool, default True
        use equal scale for x and y axis
    """


    plt.scatter(X, Y)

    if xlabel is not None:
        plt.xlabel(xlabel);

    if ylabel is not None:
        plt.ylabel(ylabel)

    # fit line through data
    m, b = np.polyfit(X, Y, 1)
    plt.plot(X, np.asarray(X)*m + b,color='k')
    if equal:
        plt.gca().set_aspect('equal')
    plt.grid()
    plt.show()


W = [70.1, 91.2, 59.5, 93.2, 53.5]
H = [1.8, 2.0, 1.7, 1.9, 1.6]
coverian_matrix=np.cov(H, W)
print(coverian_matrix)

a=np.cov(H, W, bias=1) # unbiased estimator
print(a)

#############################################
X = np.linspace(1, 10, 100)
Y = -(np.linspace(1, 5, 100) + np.sin(X)*.2)
plot_correlated_data(X, Y)
print(np.cov(X, Y))