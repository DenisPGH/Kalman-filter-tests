import filterpy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
def plot_3d_covariance(mean, cov):
    """ plots a 2x2 covariance matrix positioned at mean. mean will be plotted
    in x and y, and the probability in the z axis.
    Parameters
    ----------
    mean :  2x1 tuple-like object
        mean for x and y coordinates. For example (2.3, 7.5)
    cov : 2x2 nd.array
       the covariance matrix
    """

    # compute width and height of covariance ellipse so we can choose
    # appropriate ranges for x and y
    o, w, h = stats.covariance_ellipse(cov, 3)
    # rotate width and height to x,y axis
    wx = abs(w*np.cos(o) + h*np.sin(o)) * 1.2
    wy = abs(h*np.cos(o) - w*np.sin(o)) * 1.2


    # ensure axis are of the same size so everything is plotted with the same
    # scale
    if wx > wy:
        w = wx
    else:
        w = wy

    minx = mean[0] - w
    maxx = mean[0] + w
    miny = mean[1] - w
    maxy = mean[1] + w

    xs = np.arange(minx, maxx, (maxx-minx)/40.)
    ys = np.arange(miny, maxy, (maxy-miny)/40.)
    xv, yv = np.meshgrid(xs, ys)

    zs = np.array([100.* stats.multivariate_gaussian(np.array([x, y]), mean, cov) \
                   for x, y in zip(np.ravel(xv), np.ravel(yv))])
    zv = zs.reshape(xv.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #ax = plt.gca(projection='3d')
    ax.plot_surface(xv, yv, zv, rstride=1, cstride=1, cmap=cm.autumn)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # For unknown reasons this started failing in Jupyter notebook when
    # using `%matplotlib inline` magic. Still works fine in IPython or when
    # `%matplotlib notebook` magic is used.
    x = mean[0]
    zs = np.array([100.* stats.multivariate_gaussian(np.array([x, y]), mean, cov)
                   for _, y in zip(np.ravel(xv), np.ravel(yv))])
    zv = zs.reshape(xv.shape)

    y = mean[1]
    zs = np.array([100.* stats.multivariate_gaussian(np.array([x, y]), mean, cov)
                   for x, _ in zip(np.ravel(xv), np.ravel(yv))])
    zv = zs.reshape(xv.shape)
    plt.show()



mean = [2., 17.]
cov = [[10., 0.],
       [0., 4.]]

plot_3d_covariance(mean, cov)