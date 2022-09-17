import filterpy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import multivariate_normal


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


def plot_3d_sampled_covariance(mean, cov):
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

    count = 1000
    x, y = multivariate_normal(mean=mean, cov=cov, size=count).T

    xs = np.arange(minx, maxx, (maxx-minx)/40.)
    ys = np.arange(miny, maxy, (maxy-miny)/40.)
    xv, yv = np.meshgrid(xs, ys)

    zs = np.array([100.* stats.multivariate_gaussian(np.array([xx, yy]), mean, cov) \
                   for xx, yy in zip(np.ravel(xv), np.ravel(yv))])
    zv = zs.reshape(xv.shape)

    ax = plt.gcf().add_subplot(111, projection='3d')
    ax.scatter(x, y, [0]*count, marker='.')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    x = mean[0]
    zs = np.array([100.* stats.multivariate_gaussian(np.array([x, y]), mean, cov)
                   for _, y in zip(np.ravel(xv), np.ravel(yv))])
    zv = zs.reshape(xv.shape)
    ax.contour(xv, yv, zv, zdir='x', offset=minx-1, cmap=cm.binary)

    y = mean[1]
    zs = np.array([100.* stats.multivariate_gaussian(np.array([x, y]), mean, cov)
                   for x, _ in zip(np.ravel(xv), np.ravel(yv))])
    zv = zs.reshape(xv.shape)
    ax.contour(xv, yv, zv, zdir='y', offset=maxy, cmap=cm.binary)
    plt.grid()
    plt.show()

def plot_3_covariances():
    P = [[2, 0], [0, 2]]
    plt.subplot(131)
    plt.gca().grid(b=False)
    plt.gca().set_xticks([0, 1, 2, 3, 4])
    plot_covariance_ellipse((2, 7), cov=P, facecolor='g', alpha=0.2,
                            title='|2 0|\n|0 2|', std=[3], axis_equal=False)
    plt.ylim((0, 15))
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplot(132)
    plt.gca().grid(b=False)
    plt.gca().set_xticks([0, 1, 2, 3, 4])
    P = [[2, 0], [0, 6]]
    plt.ylim((0, 15))
    plt.gca().set_aspect('equal', adjustable='box')
    plot_covariance_ellipse((2, 7), P, facecolor='g', alpha=0.2,
                            std=[3], axis_equal=False, title='|2 0|\n|0 6|')

    plt.subplot(133)
    plt.gca().grid(b=False)
    plt.gca().set_xticks([0, 1, 2, 3, 4])
    P = [[2, 1.2], [1.2, 2]]
    plt.ylim((0, 15))
    plt.gca().set_aspect('equal', adjustable='box')
    plot_covariance_ellipse((2, 7), P, facecolor='g', alpha=0.2,
                            axis_equal=False, std=[3],
                            title='|2.0 1.2|\n|1.2 2.0|')

    plt.tight_layout()
    plt.show()




mean = [2., 17.]
cov = [[10., 0.],
       [0., 4.]]

#plot_3d_covariance(mean, cov) # joint probability density function.

#####################################################################
from filterpy.stats import gaussian, multivariate_gaussian, plot_covariance_ellipse, multivariate_multiply

x = [2.5, 7.3] # probability density
mu = [2.0, 7.0] # mean of our belief
P = [[8., 0.],
     [0., 3.]] # coverian matrix , no correlation
mult=multivariate_gaussian(x, mu, P)
print(f"{mult:.4f}")

#plot_3d_sampled_covariance(mu, P) # conturs on wals are marginal probability

#plot_3_covariances()


##################################################
from filterpy.stats import plot_covariance
import matplotlib.pyplot as plt

P = [[2, 0], [0, 6]]
#plot_covariance((2, 7), P, fc='g', alpha=0.2,std=[1, 2, 3],title='|2 0|\n|0 6|')

# plt.grid()
# plt.show()

################################################################

# max_temp = [200, 250, 300, 400, 450, 500]
# lifespan = [10, 9.7, 5, 5.4, 4.3, 0.3]
#
# plt.scatter(max_temp, lifespan, s=80)
# cov = np.cov(np.vstack((max_temp, lifespan)))
# plot_covariance((np.mean(max_temp), np.mean(lifespan)), cov, fc='g',
#             alpha=0.2, axis_equal=False, std=2)
# plt.title('Engine Temperature vs Lifespan')
# plt.xlabel('Temperature (C)'); plt.ylabel('Years')
# plt.grid()
# plt.show()

################################################################################

# P = [[4, 3.9], [3.9, 4]]
#
# plot_covariance((5, 10), P, ec='k', std=[1, 2, 3])
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid()
# plt.show()

############################### Multiplication MUltivariative #######################
""" I have plotted the original estimate (prior) in a very transparent yellow, 
the radar reading in green (evidence),
 and the finale estimate (posterior) in blue."""
from filterpy.stats import multivariate_multiply
P0 = [[6, 0], [0, 6]] # prior
P1 = [[2, 1.9], [1.9, 2]] # radar radar
P2 = multivariate_multiply((10, 10), P0, (10, 10), P1)[1] # posterior in blue
# plot_covariance_ellipse((10, 10), P0, ec='k', fc='y', alpha=0.2)
# plot_covariance_ellipse((10, 10), P1, ec='k', fc='g', alpha=0.9)
# plot_covariance_ellipse((10, 10), P2, ec='k', fc='b')
# plt.grid()
# plt.show()
## with second radar####################################################
""" The only likely place for the aircraft is where the two ellipses intersect. The intersection,
 formed by multiplying the prior and measurement, is a new Gaussian. """
P3 = [[2, -1.9], [-1.9, 2.2]] # with second radar
P4 = multivariate_multiply((10, 10), P2, (10, 10), P3)[1] #result f
plot_covariance_ellipse((10, 10), P2, ec='k', fc='y', alpha=0.6)
plot_covariance_ellipse((10, 10), P3, ec='k', fc='g', alpha=0.6)
plot_covariance_ellipse((10, 10), P4, ec='k', fc='b')
plt.grid()
plt.show()
