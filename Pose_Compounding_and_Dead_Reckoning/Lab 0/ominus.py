import numpy as np
from Pose3D import Pose3D

AxC = Pose3D(np.array([[-3], [5], [(3*np.pi)/2]])) # TODO: complete this sentence to compute the pose transformation from C-Frame to A-Frame

CxA = AxC.ominus()

print("CxA =", CxA.T)