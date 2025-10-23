from math import pi, cos, sin, sqrt
import numpy as np
import scipy
from MCLocalization import MCLocalization
from DifferentialDriveSimulatedRobot import *
from Pose3D import *
from ParticleFilter import pdf

class PF_3DOF(MCLocalization):
    def __init__(self, index, kSteps, robot, Map, particles, *args):
        super().__init__(index, kSteps, robot, particles, *args)

        self.dt = 0.1  # dt is the sampling time at which we iterate the DR
        self.wheelRadius = 0.1  # wheel radius
        self.wheelBase = 0.5  # wheel base
        self.robot.pulse_x_wheelTurns = 4096  # number of pulses per wheel turn

        self.M = Map # Save the map for the observation model (Map Based Localization)

    def GetInput(self):
        """
        Get the input for the motion model.

        :return: * **uk, Qk**. uk: input vector (:math:`u_k={}^B[\Delta L~\Delta R]^T`), Qk: covariance of the input noise

        **To be completed by the student**.
        """
        
        zsk, rsk = self.robot.ReadEncoders() # get wheel encoder readings
        uk = np.array([zsk[0, 0], zsk[1, 0]]) # store as motion model input
        Qk = np.diag(np.array([0.2 ** 2, 0.02 ** 2, np.deg2rad(2) ** 2])) # noise of the input

        return uk, Qk
    
    def GetMeasurements(self):
        """ 
        Get the measurements for the observation model.

        :return: * **zf, Rf**. zf: list of measurements, Rf: list of covariance matrices of the measurements

        **To be completed by the student**. 
        """

        zf, Rf = self.robot.ReadRanges() # get readings (zf) and measurement covariance (Rf)

        return zf, Rf

    
    def SampleProcessModel(self, particle, u, Q):
        """ 
        Apply the process model to a single particle.

        **To be completed by the student**. 
        """
        xk_1 = particle
        noise = Q
        
        R = self.robot.wheelRadius # radius of robot's wheels
        pulses = self.robot.pulse_x_wheelTurns # number of pulses for a full wheel rotation
        wheel_base = self.robot.wheelBase # distance between the two wheels

        nl, nr = (u[0]), (u[1]) # uk = [nl, nr] + noise sampled from Q
        dl = nl * (2*pi*R) / pulses # left wheel displacement
        dr = nr * (2*pi*R) / pulses # right wheel displacement
        dk = (dl + dr)/2 + noise[0][0] # distance travelled by robot aka forward distance
        yaw = ((dr - dl) / wheel_base) + noise[2][0] # angle that the robot has turned
        
        uk = Pose3D(np.array([[dk], [0 + noise[1][0]], [yaw]])) # change in position (dk along x axis, 0 along y axis and yaw angle change)
        xk = xk_1.oplus(uk) # updated position

        return xk


    def ObservationModel(self, particle, z, R):
        """ 
        Compute the measurement probability of a single particle with respect to a single measurement.
        
        **To be completed by the student**. 
        """

        distance = np.linalg.norm(particle[0:2] - self.M[z[0]]) # euclidean distance between the particle and the landmark
        likelihood = pdf(mean = z[1], sigma = np.sqrt(R), x = distance) # likelihood of the measurement

        return likelihood
    

if __name__ == '__main__':

    
    M = [np.array([[-40, 5]]).T,
           np.array([[-5, 40]]).T,
           np.array([[-5, 25]]).T,
           np.array([[-3, 50]]).T,
           np.array([[-20, 3]]).T,
           np.array([[40,-40]]).T]  # feature map. Position of 2 point features in the world frame.
    

    #Simulation:
    xs0 = np.zeros((6, 1))
    kSteps = 3000
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 0)]
    robot = DifferentialDriveSimulatedRobot(xs0, M)  # instantiate the simulated robot object

    # Particle Filter
    x0 = Pose3D(np.zeros((3,1)))  # initial guess
    P0 = np.diag([1**2, 1**2, np.deg2rad(20)**2]) # Initial uncertainty, CHOSEN BY THE STUDENT
    n_particles = 200 # Number of particles, CHOSEN BY THE STUDENT

    # create array of n_particles particles distributed randomly around x0 with covariance P
    # Each particle musy be a Pose3D object!!!

    # TO BE COMPLETED BY THE STUDENT

    particles = [
        Pose3D(x0 + np.random.normal(0, P0).diagonal().reshape((3, 1)))
        for _ in range(n_particles)
    ]

    # particles is a np.array of Pose3D objects
    usk=np.array([[0.5, 0.03]]).T
    pf = PF_3DOF(index, kSteps, robot, M, particles)
    pf.LocalizationLoop(x0, usk)

    exit(0)
