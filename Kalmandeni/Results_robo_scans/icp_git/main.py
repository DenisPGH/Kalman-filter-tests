import json

from Results_robo_scans.icp_git.basicICP import icp_point_to_plane_lm, icp_point_to_plane
import numpy as np
# fileOriginal = 'original.xyz'
# deformed = 'deformed.xyz'
#
# source_points = read_file_original(fileOriginal)
# dest_points_et_normal = read_file_deformed(deformed)

with open("js.json",'r') as jso:
    testt=json.load(jso)

a =np.array(testt['1']["node_points"][:166])
b=np.array(testt['2']["node_points"][:166])

Line1=np.array([a[:,0],a[:,1]])
Line2=np.array([b[:,0],b[:,1]])

initial = np.array([[0.01], [0.05], [0.01], [0.001], [0.001], [0.001]])
initial = np.array([[1], [0], [0], [0], [0], [0]])




icp_point_to_plane(Line1,Line2,0)

#icp_point_to_point_lm(source_points,dest_points_et_normal,initial,0)

#icp_point_to_plane_lm(Line1,Line2,initial,0)