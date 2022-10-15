import json
import numpy as np

a={
    "map": {
        "1": {
            "range": [
                1,
                2,
            ],
            "theta": 0,
            "x": 0,
            "y": 0
       }
    }

}

# read scans
with open("lidar_test_UKF_2.json",'r') as jso:
    testt=json.load(jso)

data_lidar=np.array(testt['1'])[:,1]/100 # in meters


a['map']['1']['range']=list(data_lidar)
print(a)

robot_coordinates=[(0,0,0),(0,10,0),(0,30,0),(0,100,0),(0,100,90)] # (x,y,theta)




### store scans to json

with open("test_2.json",'w') as file:
    file.write(json.dumps(a))