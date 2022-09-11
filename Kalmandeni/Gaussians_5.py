from scipy.stats import norm
import filterpy.stats
import numpy as np
print(norm(2, 3).pdf(1.5))
print(filterpy.stats.gaussian(x=1.5, mean=2, var=3*3))
###

n23 = norm(2, 3)
print('pdf of 1.5 is       %.4f' % n23.pdf(1.5))
print('pdf of 2.5 is also  %.4f' % n23.pdf(2.5))
print('pdf of 2 is         %.4f' % n23.pdf(2))
###############################################
np.set_printoptions(precision=3, linewidth=50)
print(n23.rvs(size=15))

print('variance is', n23.var())
print('standard deviation is', n23.std())
print('mean is', n23.mean())
