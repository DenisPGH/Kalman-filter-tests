import numpy as np
#import matplotlib as plt
from matplotlib import pyplot as plt

# smile=[[0.5,2],[-0.5,2],[0,0],[-0.5,0],[0.5,0],[-1,0],[1,0],[-2,0.5],[2,0.5],[-1.5,0.25],[1.5,0.25]]
# ang=90.9
# to_x=0
# to_y=0
# color='green'
# point_size=30
# dimensions_world_frame=6
# world_frame = np.array([
#     [0, 0],
#     [0, dimensions_world_frame],
#     [dimensions_world_frame, 0],
#     [dimensions_world_frame,dimensions_world_frame],
#     [dimensions_world_frame,-dimensions_world_frame],
#     [-dimensions_world_frame,-dimensions_world_frame],
#     [-dimensions_world_frame,dimensions_world_frame],
#     [-dimensions_world_frame,0],
#     [0,-dimensions_world_frame],
#
# ])
# new_places_=[1,2]
# data=np.array(smile)
# data_transf=np.array(new_places_)
# x,y=data.T
# x_t,y_t=data_transf.T
# x_w,y_w=world_frame.T
# fig, ax = plt.subplots()
# ax.scatter(x, y, color=f'{color}', s=point_size)
# ax.scatter(x_t, y_t, color=f'red', s=point_size)
# ax.scatter(0, 0, color='black', s=point_size)
# ax.scatter(x_w, y_w, color='black', s=point_size)
# #ax.annotate('transpose', (x_t, y_t),fontsize=10,color='red')
# plt.grid()
# fig.show()

class MyPlot:
    def __init__(self):
        self.plt=plt
        self.point_size=20
        self.fig=self.plt.subplots()[0]
        self.ax=self.plt.subplots()[1]

    def show_this(self,data,color):
        self.fig, self.ax = self.plt.subplots()
        data_np=np.array(data)
        x,y=data_np.T
        self.ax.scatter(x, y, color=f'{color}', s=self.point_size)
        self.fig.show()
        #elf.fig.plot()
    def end(self):
        self.fig.show()



mp=MyPlot()
mp.show_this([[2,3],[1,1]],'green')



