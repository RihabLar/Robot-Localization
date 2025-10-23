from Localization import *
import numpy as np
from DifferentialDriveSimulatedRobot import *
from Feature import *
import math

class DR_3DOFDifferentialDrive(Localization):
    """
    Dead Reckoning Localization for a Differential Drive Mobile Robot.
    """
    def __init__(self, index, kSteps, robot, x0, *args):
        """
        Constructor of the :class:`prlab.DR_3DOFDifferentialDrive` class.

        :param args: Rest of arguments to be passed to the parent constructor
        """

        super().__init__(index, kSteps, robot, x0, *args)  # call parent constructor

        self.dt = 0.1  # dt is the sampling time at which we iterate the DR
        self.t_1 = 0.0  # t_1 is the previous time at which we iterated the DR
        self.wheelRadius = 0.1  # wheel radius
        self.wheelBase = 0.5  # wheel base
        self.robot.pulse_x_wheelTurns = 4096  # number of pulses per wheel turn
        self.Forwadv_Angularv  = self.robot.pulse_x_wheelTurns*self.dt/(2*np.pi*self.wheelRadius) * np.array([[1, -self.wheelBase/2], [1, self.wheelBase/2]])

    def Localize(self, xk_1, uk):  # motion model
        """
        Motion model for the 3DOF (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`) Differential Drive Mobile robot using as input the readings of the wheel encoders (:math:`u_k=[n_L~n_R]^T`).

        :parameter xk_1: previous robot pose estimate (:math:`x_{k-1}=[x_{k-1}~y_{k-1}~\psi_{k-1}]^T`)
        :parameter uk: input vector (:math:`u_k=[u_{k}~v_{k}~w_{k}~r_{k}]^T`)
        :return xk: current robot pose estimate (:math:`x_k=[x_{k}~y_{k}~\psi_{k}]^T`)
        """

        # TODO: to be completed by the student
     
        # Store previous state and input for Logging purposes
        self.etak_1 = xk_1  # store previous state
        self.uk = uk  # store input

        R = self.robot.wheelRadius # radius of robot's wheels
        pulses = self.robot.pulse_x_wheelTurns # number of pulses for a full wheel rotation
        wheel_base = self.robot.wheelBase # distance between the two wheels

        nl, nr = uk[0], uk[1] # uk = [nl, nr]
        dl = nl * (2*math.pi*R) / pulses # left wheel displacement
        dr = nr * (2*math.pi*R) / pulses # right wheel displacement
        dk = (dl + dr)/2 # distance travelled by robot aka forward distance
        yaw = (dr - dl) / wheel_base # angle that the robot has turned
        
        change = Pose3D(np.array([dk, [0], yaw])) # change in position (dk along x axis, 0 along y axis and psi angle change)
        xk = xk_1.oplus(change) # updated position

        return xk


    def GetInput(self):
        """
        Get the input for the motion model. In this case, the input is the readings from both wheel encoders.

        :return: uk:  input vector (:math:`u_k=[n_L~n_R]^T`)
        """

        # TODO: to be completed by the student
        
        wheel_readings = self.robot.ReadEncoders()
        uk = np.array([wheel_readings[0][0], wheel_readings[0][1]])
        return uk
    
if __name__ == "__main__":

    # feature map. Position of 2 point features in the world frame.
    M = [CartesianFeature(np.array([[-40, 5]]).T),
           CartesianFeature(np.array([[-5, 40]]).T),
           CartesianFeature(np.array([[-5, 25]]).T),
           CartesianFeature(np.array([[-3, 50]]).T),
           CartesianFeature(np.array([[-20, 3]]).T),
           CartesianFeature(np.array([[40,-40]]).T)]  # feature map. Position of 2 point features in the world frame.

    xs0=np.zeros((6,1))   # initial simulated robot pose
    robot = DifferentialDriveSimulatedRobot(xs0, M) # instantiate the simulated robot object

    kSteps = 5000 # number of simulation steps
    xsk_1 = xs0 = np.zeros((6, 1))  # initial simulated robot pose
    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1)] # index of the state vector used for plotting

    x0=Pose3D(np.zeros((3,1)))
    dr_robot=DR_3DOFDifferentialDrive(index,kSteps,robot,x0)
    dr_robot.LocalizationLoop(x0, np.array([[0.5, 0.03]]).T)

    exit(0)