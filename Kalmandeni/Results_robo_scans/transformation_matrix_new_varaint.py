from math import cos, sin, radians
import numpy as np

def trig(angle):
  r = radians(angle)
  return cos(r), sin(r)

def transform_(point, TransformArray):
  p = np.array([point[0],point[1],point[2],1])
  new_p=np.dot(TransformArray,np.transpose(p))
  p=new_p

  return p

def transform(point, TransformArray):
  p = np.array([0,0,0,1])
  for i in range(0,len(point)):
      p[i] = point[i]
  p=np.dot(TransformArray,np.transpose(p))
  for i in range(0,len(point)):
      point[i]=p[i]
  return point


def matrix(rotation, translation):
  xC, xS = trig(rotation[0])
  yC, yS = trig(rotation[1])
  zC, zS = trig(rotation[2])
  dX = translation[0]
  dY = translation[1]
  dZ = translation[2]
  Translate_matrix = np.array([[1, 0, 0, dX],
                               [0, 1, 0, dY],
                               [0, 0, 1, dZ],
                               [0, 0, 0, 1]])
  Rotate_X_matrix = np.array([[1, 0, 0, 0],
                              [0, xC, -xS, 0],
                              [0, xS, xC, 0],
                              [0, 0, 0, 1]])
  Rotate_Y_matrix = np.array([[yC, 0, yS, 0],
                              [0, 1, 0, 0],
                              [-yS, 0, yC, 0],
                              [0, 0, 0, 1]])
  Rotate_Z_matrix = np.array([[zC, -zS, 0, 0],
                              [zS, zC, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
  return np.dot(Rotate_Z_matrix,np.dot(Rotate_Y_matrix,np.dot(Rotate_X_matrix,Translate_matrix)))









if __name__ == '__main__':
  point_ = [0, 0, 0]
  rotation = (0, 0, 0) # z samo
  translation = (10, 0, 0)
  mat= matrix(rotation, translation)
  print(mat)
  print(transform(point_, mat))