import math

from Results_robo_scans.UKF_DENI_Class import UKFDeni
import json
import numpy as np
import time

deni=UKFDeni()

############################# get data from the file
with open("js.json",'r') as jso:
    points=json.load(jso)
landmarks=[l['node_points'] for l in points.values()]
lm=[]
for step in landmarks[:1]:
    lm.extend([[y,x] for x,y in step])
landmarks=np.array(lm)

#################### run filter ######################

dist=100
ang=45
start=time.time()
deni.localization(0,0,0,ang,dist,landmarks, sigma_vel=0.1, sigma_steer=np.radians(1),
    sigma_range=0.3, sigma_bearing=0.1, step=1,ellipse_step=10)

print(f"x: {deni.end_x} , y: {deni.end_y} , Theta_2pi: {deni.end_theta}")
print(f"Time: {time.time()-start}")


