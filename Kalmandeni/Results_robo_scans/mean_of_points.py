import json
import matplotlib.pyplot as plt
import numpy as np

def check_point_lie_on_line(points):
    if len(points) <2:
        return False
    x1 = points[0][0]
    y1 = points[0][1]
    slope=False
    for i in range(len(points)):
        x2 = points[i][0]
        y2 = points[i][1]
        if x2 - x1 == 0:
            return False
        if slope == False:
            slope = (y2 - y1) / (x2 - x1)
            continue
        slope2 = (y2 - y1) / (x2 - x1)
        if slope != slope2:
            return False
    return True



with open("js.json",'r') as jso:
    points=json.load(jso)
all_p=[l['node_points'] for l in points.values()]
coord_robot=np.array([l["coord_of_node"][0] for l in points.values()])
lm=[]

for step in all_p[:1]:
    lm.extend([[y,x] for x,y in step[20:30]])
all_points=np.array(lm)

mmm=all_points.mean()
print(mmm)
all_points=np.array([[0,0],[1,1],])


plt.plot(all_points[:, 1], all_points[:, 0],c='green',label='all points') # all detected points
plt.axis('equal')
plt.title("Test mean")
plt.legend(loc='upper left')
plt.grid()
plt.show()

########################################################











