
import numpy as np
import re
from numpy.linalg import pinv
import transformations as transform



def read_file_original(file_path):
    a = []
    with open(file_path) as f:
        content = f.readlines()
        for line in content:
            x = float(re.split('\s+', line)[0])
            y = float(re.split('\s+', line)[1])
            z = float(re.split('\s+', line)[2])
            
            b = np.array([x,y,z])
            a.append(b)
            
    data = np.array(a)
    return data
    
    
def read_file_deformed(file_path):
    a = []
    with open(file_path) as f:
        content = f.readlines()
        for line in content:
            x = float(re.split('\s+', line)[0])
            y = float(re.split('\s+', line)[1])
            z = float(re.split('\s+', line)[2])
            
            nx = float(re.split('\s+', line)[3])
            ny = float(re.split('\s+', line)[4])
            nz = float(re.split('\s+', line)[5])
            
            b = np.array([x,y,z,nx,ny,nz])
            a.append(b)
                 
            
    data = np.array(a)
    return data
    

def icp_point_to_plane(source_points, dest_points,loop):
    """
    Point to plane matching using least squares
    source_points:  nx3 matrix of n 3D points
    dest_points: nx6 matrix of n 3D points + 3 normal vectors, which have been obtained by some
     rigid deformation of 'source_points'
    """
    
    A = []
    b = []
    
    for i in range (0,dest_points.shape[0]-1):
        
        #print( dest_points[i][3],dest_points[i][4],dest_points[i][5])
        dx = dest_points[i][0]
        dy = dest_points[i][1]
        dz = dest_points[i][2]
        nx = dest_points[i][3]
        ny = dest_points[i][4]
        nz = dest_points[i][5]

        
        sx = source_points[i][0]
        sy = source_points[i][1]
        sz = source_points[i][2]
        
        _a1 = (nz*sy) - (ny*sz)
        _a2 = (nx*sz) - (nz*sx)
        _a3 = (ny*sx) - (nx*sy)
        
        _a = np.array([_a1, _a2, _a3, nx, ny, nz])
        
        _b = (nx*dx) + (ny*dy) + (nz*dz) - (nx*sx) - (ny*sy) - (nz*sz)
        
        A.append(_a)
        b.append(_b)
        
        
    
    A1 = np.array(A)
    b1 = np.array(b)
    

    A_ = np.linalg.pinv(A1)    
    
    tr = np.dot(A_,b)
    #print(tr)

    #print(pinv(tr))
    
    #print( str(tr[0])+','+str(tr[1])+','+str(tr[2])+','+str(tr[3])+','+str(tr[4])+','+str(tr[5]))
    #print(tr[0])
    
    R = transform.euler_matrix(tr[0],tr[1],tr[2])
    R[0,3] = tr[3]
    R[1,3] = tr[4]
    R[2,3] = tr[5]
    source_transformed = []
    for i in range (0,dest_points.shape[0]-1):
        ss = np.array([(source_points[i][0]),(source_points[i][1]),(source_points[i][2]),(1)])
        p = np.dot(R,ss)
        source_transformed.append(p)
    source_points = np.array(source_transformed)
    #print(source_points)
    loop = loop + 1
    if(loop < 1):   #although this should converge in one step (which it does), you might want to reiterate over and over, just for the fun of it!
        icp_point_to_plane(source_points,dest_points,loop)
    return tr
                
        
def icp_point_to_point_lm(source_points, dest_points,initial,loop):
    """
    Point to point matching using Gauss-Newton
    
    source_points:  nx3 matrix of n 3D points

    dest_points: nx3 matrix of n 3D points, which have been obtained by some rigid deformation
    of 'source_points'

    initial: 1x6 matrix, denoting alpha, beta, gamma (the Euler angles for rotation and tx, ty, tz
    (the translation along three axis). this is the initial estimate of the transformation
    between 'source_points' and 'dest_points'

    loop: start with zero, to keep track of the number of times it loops, just a very crude way to
     control the recursion
                
    """
    
    J = []
    e = []
    
    for i in range (0,dest_points.shape[0]-1):
        
        #print('a',dest_points[i][3],dest_points[i][4],dest_points[i][5])
        #print(dest_points[i][0])
        dx = dest_points[i][0]
        dy = dest_points[i][1]
        dz = dest_points[i][2]
        
        sx = source_points[i][0]
        sy = source_points[i][1]
        sz = source_points[i][2]
        
        alpha = initial[0][0]
        beta = initial[1][0]
        gamma = initial[2][0]
        tx = initial[3][0]        
        ty = initial[4][0]
        tz = initial[5][0]
        #print alpha
        
        a1 = (-2*beta*sx*sy) - (2*gamma*sx*sz) + (2*alpha*((sy*sy) + (sz*sz))) + (2*((sz*dy) - (sy*dz))) + 2*((sy*tz) - (sz*ty))
        a2 = (-2*alpha*sx*sy) - (2*gamma*sy*sz) + (2*beta*((sx*sx) + (sz*sz))) + (2*((sx*dz) - (sz*dx))) + 2*((sz*tx) - (sx*tz))
        a3 = (-2*alpha*sx*sz) - (2*beta*sy*sz) + (2*gamma*((sx*sx) + (sy*sy))) + (2*((sy*dx) - (sx*dy))) + 2*((sx*ty) - (sy*tx))
        a4 = 2*(sx - (gamma*sy) + (beta*sz) +tx -dx)
        a5 = 2*(sy - (alpha*sz) + (gamma*sx) +ty -dy)
        a6 = 2*(sz - (beta*sx) + (alpha*sy) +tz -dz)
        
        _residual = (a4*a4/4)+(a5*a5/4)+(a6*a6/4)
        
        _J = np.array([a1, a2, a3, a4, a5, a6])
        #print(_J)
        _e = np.array([_residual]) # error

        
        J.append(_J)
        e.append(_e)
        
    jacobian = np.array(J)
    #print(jacobian)
    residual = np.array(e)
    print(residual)
    
    update = -np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(jacobian),jacobian)),np.transpose(jacobian)),residual)
    #print(update)
    #print( update, initial)
    
    initial = initial + update
    #print(initial)

    
    #print( np.transpose(initial))
    
    loop = loop + 1
    
    if(loop < 2):  # here lies the control variable, control the number of iteration from here
    
        icp_point_to_point_lm(source_points,dest_points,initial, loop)
        
        
def icp_point_to_plane_lm(source_points, dest_points,initial,loop):
    """
    Point to plane matching using Gauss Newton
    
    source_points:  nx3 matrix of n 3D points
    dest_points: nx6 matrix of n 3D points + 3 normal vectors, which have been obtained by some rigid deformation of 'source_points'
    initial: 1x6 matrix, denoting alpha, beta, gamma (the Euler angles for rotation and tx, ty, tz (the translation along three axis). 
                this is the initial estimate of the transformation between 'source_points' and 'dest_points'
    loop: start with zero, to keep track of the number of times it loops, just a very crude way to control the recursion            
                
    """
    
    J = []
    e = []
    
    for i in range (0,dest_points.shape[0]-1):
        
        #print dest_points[i][3],dest_points[i][4],dest_points[i][5]
        dx = dest_points[i][0]
        dy = dest_points[i][1]
        dz = dest_points[i][2]
        nx = dest_points[i][3]
        ny = dest_points[i][4]
        nz = dest_points[i][5]
        
        sx = source_points[i][0]
        sy = source_points[i][1]
        sz = source_points[i][2]
        
        alpha = initial[0][0]
        beta = initial[1][0]
        gamma = initial[2][0]
        tx = initial[3][0]        
        ty = initial[4][0]
        tz = initial[5][0]
        
        a1 = (nz*sy) - (ny*sz)
        a2 = (nx*sz) - (nz*sx)
        a3 = (ny*sx) - (nx*sy)
        a4 = nx
        a5 = ny
        a6 = nz
        
        _residual = (alpha*a1) + (beta*a2) + (gamma*a3) + (nx*tx) + (ny*ty) + (nz*tz) - (((nx*dx) + (ny*dy) + (nz*dz)) - ((nx*sx) + (ny*sy) + (nz*sz)))
    
        _J = np.array([a1, a2, a3, a4, a5, a6])
        _e = np.array([_residual])
        
    J.append(_J)
    e.append(_e)
        
    jacobian = np.array(J)
    residual = np.array(e)
    
    update = -np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(jacobian),jacobian)),np.transpose(jacobian)),residual)
    
    #print update, initial
    
    initial = initial + update
    #print(initial)
    
    print('a',np.transpose(initial))
    
    loop = loop + 1
    
    if(loop < 50):  # here lies the control variable, control the number of iteration from here
    
        icp_point_to_point_lm(source_points,dest_points,initial, loop)








