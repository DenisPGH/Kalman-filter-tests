from matplotlib import pyplot as plt
from numpy.random import randn
plt.plot([randn() for _ in range(3000)], lw=1)
plt.show()