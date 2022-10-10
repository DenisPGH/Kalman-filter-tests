import json

from Results_robo_scans.icp_git.basicICP import icp_point_to_plane_lm, icp_point_to_plane, icp_point_to_point_lm
import numpy as np
import matplotlib.pyplot as plt
# fileOriginal = 'original.xyz'
# deformed = 'deformed.xyz'
#
# source_points = read_file_original(fileOriginal)
# dest_points_et_normal = read_file_deformed(deformed)
from Results_robo_scans.transofmation_vector_ukf import DeniTransformation

with open("js.json",'r') as jso:
    testt=json.load(jso)

a =np.array(testt['1']["node_points"])
b=np.array(testt['5']["node_points"])
a=np.insert(a, 2, 0, axis=1) # add a Z axis=0

b=np.insert(b, 2, 0, axis=1) # add a Z axis=0
b=np.insert(b, 3, 0, axis=1) # add a Z axis=0
b=np.insert(b, 4, 0, axis=1) # add a Z axis=0
b=np.insert(b, 5, 0, axis=1) # add a Z axis=0
#print(b)
# b=np.insert(b, 3, 0, axis=0) # add a Z axis=0
# b=np.insert(b, 4, 0, axis=1) # add a Z axis=0
# b=np.insert(b, 5, 0, axis=1) # add a Z axis=0

Line1=np.array([a[:,0],a[:,1],a[:,2]])
#Line2=np.array([b[:,0],b[:,1],b[:,2]])
Line2=np.array([b[:,0],b[:,1],b[:,2],b[:,3],b[:,4],b[:,5]])


#initial = np.array([[0.01], [0.05], [0.01], [0.001], [0.001], [0.001]])
initial = np.array([[0], [0], [0], [0], [0], [0]]) # start position




icp_point_to_plane(Line2,Line1,0)
print("dddddddddddddddddddddddddddddddd")

#icp_point_to_point_lm(Line1,Line2,initial,0)
print("dddddddddddddddddddddddddddddddd")
#icp_point_to_plane_lm(Line1,Line2,initial,0)

#################

a=np.delete(a,2,axis=1) # remove Z axis

b=np.delete(b,5,axis=1) # remove Z axis
b=np.delete(b,4,axis=1) # remove Z axis
b=np.delete(b,3,axis=1) # remove Z axis
b=np.delete(b,2,axis=1) # remove Z axis



DT=DeniTransformation()
c=DT.translocation(b,13.56,0.21,0)
c=np.array(c)
# x,y=data.T
# x_t,y_t=data_transf.T
# x_w,y_w=world_frame.T

plt.scatter(a[:,0], a[:,1], color=f'red', s=5,label='first scan')
plt.scatter(b[:,0], b[:,1], color=f'green', s=5,label='second scan')
plt.scatter(c[:,0], c[:,1], color=f'blue', s=5,label='transformed')
plt.legend(loc='upper left')
plt.grid()
plt.show()
