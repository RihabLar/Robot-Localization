from GFLocalization import *
from EKF import *
from DR_3DOFDifferentialDrive import *
from DifferentialDriveSimulatedRobot import *
from MapFeature import *
import scipy.linalg

class EKF_3DOFDifferentialDriveCtVelocity(GFLocalization, DR_3DOFDifferentialDrive, EKF):

    def __init__(self, kSteps, robot, *args):
        x0 = np.zeros((6, 1))  # initial state x0=[x y yaw u v yaw_dot]^T
        P0 = np.zeros((6, 6))  # initial covariance

        # Define necessary parameters
        self.dt = 0.1  # time step
        self.wheelRadius = 0.1  # radius of the wheels
        self.wheelBase = 0.5  # distance between the two wheels

        # This is required for plotting
        index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1),
                 IndexStruct("u", 3, 2), IndexStruct("v", 4, 3), IndexStruct("yaw_dot", 5, None)]

        self.t_1 = 0
        self.t = 0
        self.Dt = self.t - self.t_1
        super().__init__(index, kSteps, robot, x0, P0, *args)

    def f(self, xk_1, uk):
        #return the predicted state vector
        vk_bar = xk_1[3:6].reshape((3, 1))
        uk = vk_bar * self.dt
        etak_bar = Pose3D.oplus(xk_1[0:3].reshape((3, 1)), uk.reshape((3, 1)))
        xk_bar = np.block([[etak_bar], [vk_bar]])
        return xk_bar

    def Jfx(self, xk_1):
        #return the Jacobian matrix
        uk = xk_1[3:6].reshape((3, 1)) * self.dt
        J_eta_x = np.block([Pose3D.J_1oplus(xk_1[0:3].reshape((3, 1)), uk.reshape((3, 1))), Pose3D.J_2oplus(xk_1[0:3].reshape((3, 1))) * self.dt])
        J_v_x = np.block([np.zeros((3, 3)), np.diag(np.ones(3))])
        J = np.block([[J_eta_x], [J_v_x]])
        return J

    def Jfw(self, xk_1):
        #compute the Jacobian of the process noise with respect to the state vector
        J = np.block([[Pose3D.J_2oplus(xk_1[0:3].reshape((3, 1))) * self.dt * self.dt / 2], [np.diag(np.ones(3)) * self.dt]])
        return J

    def Jhx(self):
        #define the measurement Jacobian matrix H_k
        H_k = np.array([
            [0, 0, 1, 0, 0, 0],  # yaw
            [0, 0, 0, 1, 0, 0],  # u 
            [0, 0, 0, 0, 0, 1]   # yaw dot 
        ])
        return H_k

    def h(self, xk):  #:hm(self, xk):
     
            H = np.zeros((0, 6))
            # if the heading measurement is valid, add the corresponding row to H
            if self.heading_val == True:
                H_k = self.Jhx()
                H_compass = H_k[0].reshape(1, -1)  # Take the first row for yaw
                H = np.block([[H], [H_compass]])
            # if the encoder measurements are valid, add the corresponding rows to H
            if self.encoder_val == True:
                H_k = self.Jhx()
                H_encoder = np.zeros((2, 6))
                H_encoder[0, 3] = H_k[1, 3] * self.Forwadv_Angularv[0, 0]
                H_encoder[0, 5] = H_k[1, 3] * self.Forwadv_Angularv[0, 1]
                H_encoder[1, 3] = H_k[2, 5] * self.Forwadv_Angularv[1, 0]
                H_encoder [1, 5] = H_k[2, 5] * self.Forwadv_Angularv[1, 1]
                H = np.block([[H], [H_encoder ]])

            h = H @ xk.reshape((6, 1))
            #return the predicted measurement
            return h
    
    def GetInput(self):
        uk = None
        W = np.array([0.5 ** 2, 0.01 ** 2, np.deg2rad(1) ** 2])         
        Qk = np.diag(W)      
        #return the input (uk) and the process noise covariance matrix (Qk)                                              
        return uk, Qk
    
    def GetMeasurements(self):
        zk, Rk = np.zeros((0, 1)), np.zeros((0, 0))
        Hk, Vk = np.zeros((0, 6)), np.zeros((0, 0))
        
        self.heading_val  = False

        # Read compass (yaw) measurement
        z_compass, sigma2_compass = self.robot.ReadCompass()
        if z_compass.size > 0:
            zk, Rk = np.vstack([zk, z_compass]), scipy.linalg.block_diag(Rk, sigma2_compass)
            Hk, Vk = np.vstack([Hk, self.Jhx()[0].reshape(1, -1)]), scipy.linalg.block_diag(Vk, np.eye(1))
            self.heading_val = True

        # Read encoder measurements
        z_encoder, R_encoder= self.robot.ReadEncoders()
        if z_encoder.size > 0:
            H_encoder = np.zeros((2, 6))
            H = self.Jhx()
            H_encoder[0, 3] = H[1, 3] * self.Forwadv_Angularv[0, 0]
            H_encoder[1, 5] = H[2, 5] * self.Forwadv_Angularv[1, 1]

            zk, Rk = np.vstack([zk, z_encoder]), scipy.linalg.block_diag(Rk, R_encoder)
            Hk, Vk = np.vstack([Hk, H_encoder]), scipy.linalg.block_diag(Vk, np.eye(2))
            self.encoder_val = True
        #return the measurements and their covariances, or empty arrays if no measurements are available
        return zk, Rk, Hk, Vk if zk.shape[0] > 0 else (np.zeros((0, 1)), np.zeros((0, 0)), np.zeros((1, 0)), np.zeros((0, 0)))


if __name__ == '__main__':
    M = [CartesianFeature(np.array([[-40, 5]]).T),
         CartesianFeature(np.array([[-5, 40]]).T),
         CartesianFeature(np.array([[-5, 25]]).T),
         CartesianFeature(np.array([[-3, 50]]).T),
         CartesianFeature(np.array([[-20, 3]]).T),
         CartesianFeature(np.array([[40, -40]]).T)]

    xs0 = np.zeros((6, 1))
    robot = DifferentialDriveSimulatedRobot(xs0, M)
    kSteps = 5000

    index = [IndexStruct("x", 0, None), IndexStruct("y", 1, None), IndexStruct("yaw", 2, 1),
             IndexStruct("u", 3, 2), IndexStruct("v", 4, 3), IndexStruct("yaw_dot", 5, None)]

    x0 = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
    P0 = np.diag(np.array([0.0, 0.0, 0.0, 0.5 ** 2, 0 ** 2, 0.05 ** 2]))

    dd_robot = EKF_3DOFDifferentialDriveCtVelocity(kSteps, robot)
    dd_robot.LocalizationLoop(x0, P0, np.array([[0.5, 0.03]]).T)

    exit(0)
