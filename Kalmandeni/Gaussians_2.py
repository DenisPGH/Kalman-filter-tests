import matplotlib.pyplot as plt
import numpy as np
import math
from filterpy.stats import gaussian
from filterpy.stats import plot_gaussian_pdf
x = np.arange(-3, 3, .01)
print(x)
#plt.plot(x, np.exp(-x**2))
#plt.show()

def gaussian_(x, mean, var, normed=True):
    """
    returns probability density function (pdf) for x given a Gaussian with the
    specified mean and variance. All must be scalars.
    gaussian (1,2,3) is equivalent to scipy.stats.norm(2, math.sqrt(3)).pdf(1)
    It is quite a bit faster albeit much less flexible than the latter.
    Parameters
    ----------
    x : scalar or array-like
        The value(s) for which we compute the distribution
    mean : scalar
        Mean of the Gaussian
    var : scalar
        Variance of the Gaussian
    normed : bool, default True
        Normalize the output if the input is an array of values.
    Returns
    -------
    pdf : float
        probability distribution of x for the Gaussian (mean, var). E.g. 0.101 denotes
        10.1%.
    Examples
    --------
    #>>> gaussian(8, 1, 2)
    1.3498566943461957e-06
    #>>> gaussian([8, 7, 9], 1, 2)
    array([1.34985669e-06, 3.48132630e-05, 3.17455867e-08])
    """

    pdf = ((2*math.pi*var)**-.5) * np.exp((-0.5*(np.asarray(x)-mean)**2.) / var)
    if normed and len(np.shape(pdf)) > 0:
        pdf = pdf / sum(pdf)

    return pdf

plot_gaussian_pdf(22, 4, mean_line=True, xlabel='$^{\circ}C$')
plt.show()


######

from filterpy.stats import norm_cdf
print('Cumulative probability of range 21.5 to 22.5 is {:.2f}%'.format(
      norm_cdf((21.5, 22.5), 22,4)*100))
print('Cumulative probability of range 23.5 to 24.5 is {:.2f}%'.format(
      norm_cdf((23.5, 24.5), 22,4)*100))

