import matplotlib.pyplot as plt
import numpy as np
import math
from filterpy.stats import gaussian
#### Sum of normally distributed random variables
#### #####################################
#### Sum of normally distributed random variables
x = np.arange(-1, 3, 0.01)
g1 = gaussian(x, mean=0.8, var=.1)
g2 = gaussian(x, mean=1.3, var=.2)
plt.plot(x, g1, x, g2)

g = g1 * g2  # element-wise multiplication
g = g / sum(g)  # normalize
plt.plot(x, g, ls='-.')
plt.show()