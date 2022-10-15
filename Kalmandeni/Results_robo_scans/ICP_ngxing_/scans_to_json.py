import json
import numpy as np

# a={
#     "map": {
#         "1": {
#             "range": [
#                 1,
#                 2,
#             ],
#             "theta": 0,
#             "x": 0,
#             "y": 0
#        }
#     }
#
# }

# read scans
with open("lidar_test_UKF_2.json",'r') as jso:
    testt=json.load(jso)


# "theta": 0, "x": 0, "y": 0}
counter=1
a={}
a['map']={}
rc=[(0,0,0),(0,10,0),(0,30,0),(0,100,0),(0,100,90)] # (x,y,theta)
cr=0
for steps,info in testt.items():
    if cr>1:
        break
    data_lidar = np.array(testt[f'{counter}'])[:, 1] / 100  # in meters
    a['map'][f'{counter}']={}
    a['map'][f'{counter}']['range'] = list(data_lidar[:200])
    a['map'][f'{counter}']['theta'] =rc[counter-1][2]
    a['map'][f'{counter}']['x'] =rc[counter-1][0]/100
    a['map'][f'{counter}']['y'] =rc[counter-1][1]/100
    counter+=1
    cr+=1



print(a)


### store scans to json

with open("test_2.json",'w') as file:
    file.write(json.dumps(a))