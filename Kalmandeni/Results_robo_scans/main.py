import math

from Results_robo_scans.UKF_DENI_Class import UKFDeni
import json
import numpy as np
import time
import matplotlib.pyplot as plt

deni=UKFDeni()

############################# get data from the file
# coord_robot=[]
all_points=[]
with open("js.json",'r') as jso:
    points=json.load(jso)
all_p=[l['node_points'] for l in points.values()]
coord_robot=np.array([l["coord_of_node"][0] for l in points.values()])
lm=[]
for step in all_p:
    lm.extend([[y,x] for x,y in step])
all_points=np.array(lm)
landmarks=np.array([[y,x] for x,y in all_p[0]])

#################### run filter ######################
test_path=[(0,0),(0,20),(0,80),(0,20),(0,40),
            (90,40),(90,30),(90,50),(90,60),(90,10),
            (180,20),(180,50),(180,50),(180,50)
            ] # (dir,dist)



start=time.time()
full_path=[]
for dir,dist in test_path:
    deni.localization(deni.start_x,deni.start_y,deni.start_theta,dir,dist,landmarks, sigma_vel=0.1, sigma_steer=np.radians(1),
        sigma_range=0.3, sigma_bearing=0.1, step=1,ellipse_step=10)
    full_path.extend(deni.track)

full_path=np.array(full_path)

############ visualisation ###############################


plt.scatter(landmarks[:, 1], landmarks[:, 0], s=30,c='yellow') # only landmark,which I use in filter
plt.scatter(all_points[:, 1], all_points[:, 0], s=10,c='green') # all detected points
plt.plot(full_path[:, 1], full_path[:, 0], color='k', lw=2) # path from kalman filter
plt.plot(coord_robot[:, 0], coord_robot[:, 1], color='blue', lw=2) # path robot from other sensors
plt.axis('equal')
plt.title("UKF Robot localization")
plt.grid()
plt.show()


print(f"x: {deni.end_x} , y: {deni.end_y} , Theta_2pi: {deni.end_theta}")
print(f"Time: {time.time()-start}")


