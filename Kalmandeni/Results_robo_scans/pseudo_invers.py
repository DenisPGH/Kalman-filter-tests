import numpy as np
from numpy.linalg import inv

def translocation( coordinates, new_x, new_y, teta):

    end = np.insert(coordinates, 2, 1, axis=1)  # add a Z axis
    teta = np.radians(teta)
    trans_matrix = np.array([
        [np.cos(teta), np.sin(teta), 0],
        [-np.sin(teta), np.cos(teta), 0],
        [new_x, new_y, 1]])
    result_homog_matrix = np.dot(end, trans_matrix)
    result_end = np.delete(result_homog_matrix, 2, axis=1)  # remove Z axis
    return result_end


res=translocation(np.array([(0,0)]),10,0,0)
print(res)
#print(inv(res))
# new_x=10
# new_y=0
# teta=0
# trans_matrix = np.array([
#         [np.cos(teta), np.sin(teta), 0],
#         [-np.sin(teta), np.cos(teta), 0],
#         [-new_x, -new_y, 1]])
# start_point=np.array([0,0,0])
# end = np.insert(start_point, 2, 1, axis=1) # add a Z axis
# res=np.dot(end,trans_matrix)
# print(res)
