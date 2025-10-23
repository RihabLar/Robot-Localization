import numpy as np
from Pose3D import Pose3D
from math import atan2, cos, sin

AxB = Pose3D(np.array([[1], [2], [np.pi/2]])) # TODO: complete this sentence to define the Pose Transformation as defined above
BxC = Pose3D(np.array([[3], [4], [np.pi]])) # TODO: complete this sentence to  define the Pose Transformation as defined above

AxC = AxB.oplus(BxC) # TODO: complete this sentence to compute the pose transformation from A-Frame to C-Frame

print("AxC =", AxC.T)