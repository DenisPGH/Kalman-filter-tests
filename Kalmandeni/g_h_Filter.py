import matplotlib.pylab as pylab
import numpy as np
from matplotlib import pyplot as plt

from ploting import MyPlot


def g_h_filter(data, x0, dx, g, h, dt=1.):
    """
       Performs g-h filter on 1 state variable with a fixed g and h.

       'data' contains the data to be filtered.
       'x0' is the initial value for our state variable
       'dx' is the initial change rate for our state variable
       'g' is the g-h's g scale factor
       'h' is the g-h's h scale factor
       'dt' is the length of the time step
       """
    x_est = x0
    results = []
    for z in data:
        # prediction step
        x_pred = x_est + (dx*dt)
        dx = dx

        # update step
        residual = z - x_pred
        dx = dx + h * (residual) / dt
        x_est = x_pred + g * residual
        results.append(x_est)
    return np.array(results)

weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]


mp=MyPlot()

#book_plots.plot_track([0, 11], [160, 172], label='Actual weight')
data = g_h_filter(data=weights, x0=160., dx=1., g=6./10, h=2./3, dt=1.)
#plot_g_h_results(weights, data)
print(weights)
print(data)
#visual
start_kg,end_kg=160,172
weights=[[x,e] for x,e in enumerate(weights)]
data=[[x,e] for x,e in enumerate(data)]
state=np.array([[0,start_kg],[len(weights)-1,end_kg]])

color_w='black'
color_d='blue'
point_size=20
fig, ax = plt.subplots()
weight_np=np.array(weights)
data_np=np.array(data)
x,y=weight_np.T
x_d,y_d=data_np.T
x_s,y_s=state.T
ax.scatter(x, y, color=f'{color_w}', s=point_size)
ax.scatter(x_d, y_d, color=f'{color_d}', s=point_size)
ax.plot(x_d, y_d, color=f'{color_d}')
ax.plot(x_s,y_s , color=f'yellow')
plt.grid()
fig.show()
print(weights[0])
