from DifferentialDriveSimulatedRobot import *
from DR_3DOFDifferentialDrive import *

if __name__ == "__main__":

    # feature map. Position of 2 point features in the world frame.
    M2D = [np.array([[-40, 5]]).T,
           np.array([[-5, 40]]).T,
           np.array([[-5, 25]]).T,
           np.array([[-3, 50]]).T,
           np.array([[-20, 3]]).T,
           np.array([[40,-40]]).T]
    xs0 = np.zeros((6,1))   # initial simulated robot pose

    robot = DifferentialDriveSimulatedRobot(xs0, M2D) # instantiate the simulated robot object

    kSteps = 1250 # number of simulation steps
    xsk_1 = xs0 = np.zeros((6, 1))  # initial simulated robot pose
    
    # Linear and angular velocity
    usk_pos = np.array([[0.8, 0.1]]).T  
    usk_neg = np.array([[0.8, -0.1]]).T  

    # Simulate the robot's motion over k time steps
    for k in range(kSteps):
        if k > (kSteps/2):
            usk = usk_pos # Start with positive angular velocity
        else:
            usk = usk_neg # Switch to negative angular velocity

        xsk = robot.fs(xsk_1, usk) # Update current state
        xsk_1 = xsk  # Update previous state to current state

        # Visualize the robot at every n time steps
        if k % robot.visualizationInterval == 0:
                robot.PlotRobot()

    plt.show()