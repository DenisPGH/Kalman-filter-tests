import matplotlib.pyplot as plt
import numpy as np
import math
from filterpy.stats import gaussian
from filterpy.stats import plot_gaussian_pdf
#import kf_book.book_plots as book_plots
def bar_plot(pos, x=None, ylim=(0,1), title=None, c='#30a2da',
             **kwargs):
    """ plot the values in `pos` as a bar plot.
    **Parameters**
    pos : list-like
        list of values to plot as bars
    x : list-like, optional
         If provided, specifies the x value for each value in pos. If not
         provided, the first pos element is plotted at x == 0, the second
         at 1, etc.
    ylim : (lower, upper), default = (0,1)
        specifies the lower and upper limits for the y-axis
    title : str, optional
        If specified, provides a title for the plot
    c : color, default='#30a2da'
        Color for the bars
    **kwargs : keywords, optional
        extra keyword arguments passed to ax.bar()
    """

    ax = plt.gca()
    if x is None:
        x = np.arange(len(pos))
    ax.bar(x, pos, color=c, **kwargs)
    if ylim:
        plt.ylim(ylim)
    plt.xticks(np.asarray(x), x)
    if title is not None:
        plt.title(title)

def normalize(p):
    return p / sum(p)

def update_old(likelihood, prior):
    return normalize(likelihood * prior)

def update(likelihood, prior):
    posterior = prior * likelihood   # p(z|x) * p(x) Bayes theorem
    return normalize(posterior)

prior =      normalize(np.array([4, 2, 0, 7, 2, 12, 35, 20, 3, 2]))
likelihood = normalize(np.array([3, 4, 1, 4, 2, 38, 20, 18, 1, 16]))
posterior = update(likelihood, prior)
#book_plots.bar_plot(posterior)
print(posterior)
plt.plot(posterior, ls='-.')
plt.show()

xs = np.arange(0, 10, .01)


def mean_var(p):
    x = np.arange(len(p))
    mean = np.sum(p * x,dtype=float)
    var = np.sum((x - mean)**2 * p)
    return mean, var

mean, var = mean_var(posterior)
bar_plot(posterior)
plt.plot(xs, gaussian(xs, mean, var, normed=False), c='r')
plt.grid()
plt.show()
print('mean: %.2f' % mean, 'var: %.2f' % var)