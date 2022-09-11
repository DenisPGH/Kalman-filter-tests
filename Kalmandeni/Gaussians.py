### standart deviation
from numpy.random import randn
import numpy as np
data = 1.8 + randn(100)*.1414
mean, std = data.mean(), data.std()
print(f'mean = {mean:.3f}')
print(f'std  = {std:.3f}')
res=np.sum((data > mean-std) & (data < mean+std)) / len(data) * 100
print(f"Percent in std= {res} %")

## gausians
X = [1.8, 2.0, 1.7, 1.9, 1.6]
mean=np.mean(X)
variance=np.std(X)
# Plot Gausians visualisation
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import scipy.stats as stats

mean = 7 #3
std = 6 #2
data = random.normal(loc=mean, scale=std, size=50000)
print(len(data))
print(data.mean())
print(data.std())

def plot_normal(xs, mean, std, **kwargs):
    norm = stats.norm(mean, std)
    plt.plot(xs, norm.pdf(xs), **kwargs)
    plt.show()

xs = np.linspace(-5, 20, num=200)
plot_normal(xs, mean, std, color='k')

