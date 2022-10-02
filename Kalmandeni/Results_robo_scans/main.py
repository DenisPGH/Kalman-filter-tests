import math

from Results_robo_scans.UKF_DENI_Class import UKFDeni
import json
import numpy as np
import time
import matplotlib.pyplot as plt

deni=UKFDeni()
start=time.time()
############################# get data from the file
with open("js.json",'r') as jso:
    points=json.load(jso)
all_p=[l['node_points'] for l in points.values()]
coord_robot=np.array([l["coord_of_node"][0] for l in points.values()])
lm=[]

for step in all_p[:1]:
    lm.extend([[y,x] for x,y in step])
all_points=np.array(lm)

#landmarks=np.array([[y,x] for x,y in all_p[0]]) # [:30] control kolko landmakrs
landmarks=np.array([[100,math.radians(0)]])

land_all=[np.array([[100,math.radians(0)]]),np.array([[100,math.radians(0)]]),np.array([[15,math.radians(0)]])]

#################### run filter ######################
test_path=[(0,0),(0,20),(90,80),(0,20),(0,40),
            (90,40),(90,30),(90,50),(90,60),(90,10),
            (180,20),(180,50),(180,50),(180,50)
            ] # (dir,dist)




full_path=[]
counter=0
aa=[]
for dir,dist in test_path[:3]: # controll the steps[:7]
    a=deni.localization(deni.start_x,deni.start_y,deni.start_theta,dir,dist,land_all[counter], sigma_vel=0.5,
    sigma_steer=np.radians(1),sigma_range=200, sigma_bearing=.01, step=1,ellipse_step=10)
    full_path.extend(deni.track)
    print(a)
    counter+=1
    aa=a

full_path=np.array(full_path)

############ visualisation ###############################


#plt.scatter(landmarks[:, 1], landmarks[:, 0], s=30,c='yellow',label=f'landmarks {len(landmarks)}') # only landmark,which I use in filter
# plt.scatter(land_all[0][:, 1], land_all[0][:, 0], s=30,c='yellow',label=f'landmarks {len(landmarks)}') # only landmark,which I use in filter
# plt.scatter(land_all[1][:, 1], land_all[1][:, 0], s=30,c='red',label=f'landmarks {len(landmarks)}') # only landmark,which I use in filter
# plt.scatter(land_all[2][:, 1], land_all[2][:, 0], s=30,c='blue',label=f'landmarks {len(landmarks)}') # only landmark,which I use in filter
#plt.plot(all_points[:, 1], all_points[:, 0],c='green',label='all points') # all detected points
#plt.plot(full_path[:, 1], full_path[:, 0], color='k', lw=2,label='filter') # path from kalman filter
#plt.plot(coord_robot[:, 0], coord_robot[:, 1], color='blue', lw=2,label='path') # path robot from other sensors
plt.plot(0,400,color='red',label=f"Time: {time.time()-start:.2f}sec")
plt.plot(0,400,color='red',label=f"Orient: {deni.start_theta}")
plt.plot(0,400,color='red',label=f"X: {aa[0]}")
plt.plot(0,400,color='red',label=f"Y: {aa[1]}")
plt.axis('equal')
plt.title("UKF Robot localization")
plt.legend(loc='upper left')
plt.grid()
plt.show()



