from Localization import Localization
from ParticleFilter import ParticleFilter
import matplotlib.pyplot as plt
import numpy as np
class MCLocalization(ParticleFilter, Localization):
    """
    Monte Carlo Localization class.

    This class is used as "Dead Reckoning" localization using a Particle Filter.
    It implements the Prediction method from :class:`ParticleFilter` and the 
    Localize and LocalizationLoop methods from :class:`Localization`.
    """

    def __init__(self, index, kSteps, robot, particles, *args):
        """
        Constructor.
        :param index: Logging index structure (:class:`prpy.Index`)
        :param kSteps: Number of time steps to simulate
        :param robot: Simulation robot object (:class:`prpy.Robot`)
        :param particles: initial particles as a list of Pose objects (or at least a list of numpy arrays)
        :param args: arguments to be passed to the parent constructor
        """

        # super Localization
        ParticleFilter.__init__(self, particles, *args)
        Localization.__init__(self, index, kSteps, robot, self.get_mean_particle(), *args)

        self.robot.visualizationInterval = 20

        self.init_plotting()

    def GetMeasurements(self):
        """
        Read the measurements from the robot.
        The measurements arrive at a frequency defined in the :attr:`SimulatedRobot.SimulatedRobot.Distance_feature_reading_frequency` attribute.
        Must return two lists: z and R where
        - z is a list of measurements
        - R is a list of Covariance matrices for each measurement

        Two options are available for z:
           - each element of z is a single measurement
           - each element of z is a tuple with the first element being the feature id and the second the measurement

        The second option is used when the feature id is known, and helps to avoid the complexity of the data association.

        :returns: z, R
        """

        print("GetMeasurements must be implemented in a child class")
        pass


    def Localize(self):
        """
        Single Localization iteration. Given the previous robot pose, the function reads the input and computes the current pose.

        :returns: **xk** current robot pose (we can assume the mean of the particles or the most likely particle)

        """
        uk, Qk = self.GetInput()  # Get the input from the robot (inheritance from Localization)
        if uk.size > 0:
            self.Prediction(uk, Qk)

        zf, Rf = self.GetMeasurements() # Get the measurements from the robot (new, for map based localization)
        if len(zf) > 0:
            self.Update(zf, Rf)

        return self.get_mean_particle() # choose mean particle or best particle
    
    def LocalizationLoop(self, x0, usk):
        """
        Given an initial robot pose :math:`x_0` and the input to the :class:`prpy.SimulatedRobot` this method calls iteratively :meth:`prpy.DRLocalization.Localize` for k steps, solving the robot localization problem.

        Overwritten function to fix Localize, which must represent a single iteration
        
        :param x0: initial robot pose
        :param usk: The control input for the robot

        """
        xk_1 = x0
        xsk_1 = self.robot.xsk_1

        for self.k in range(self.kSteps):
            xsk = self.robot.fs(xsk_1, usk)  # Simulate the robot motion

            self.xk = self.Localize()  # Localize the robot

            xsk_1 = xsk  # current state becomes previous state for next iteration
            print(self.particle_weights)
            self.PlotTrajectory()  # plot the estimated trajectory
        plt.show()
        return
 
   
    '''
    Plotting
    '''
    def PlotTrajectory(self):
        """ Overwritten PlotTrajectory to include plotting each particle """
        Localization.PlotTrajectory(self)
        self.PlotParticles()

    def init_plotting(self):
        """
        Init the plotting of the particles and the mean particle.
        """
        self.x_idx = 0
        self.y_idx = 1
        self.yaw_idx = 2

        for x in self.index:
            if x.state == 'x': self.x_idx = x.simulation
            if x.state == 'y': self.y_idx = x.simulation
            if x.state == 'yaw': self.yaw_idx = x.simulation

        self.plt_particles = []
        self.plt_particles_ori = []
        for i in range(len(self.particles)):
            plt_particle, = plt.plot(self.particles[i][self.x_idx], self.particles[i][1], 'g.', markersize=2)
            # make plot on top
            plt_particle.set_zorder(10)
            self.plt_particles.append(plt_particle)

        for i in range(len(self.particles)):
            plt_particle, = plt.plot([self.particles[i][self.x_idx], self.particles[i][self.x_idx] + 0.5 * np.cos(self.particles[i][self.yaw_idx])],
                                     [self.particles[i][self.y_idx], self.particles[i][self.y_idx] + 0.5 * np.sin(self.particles[i][self.yaw_idx])], 'g',
                                     markersize=1)
            # make plot on top
            plt_particle.set_zorder(10)
            self.plt_particles_ori.append(plt_particle)
        
        # append the mean particle
        plt_mean_particle, = plt.plot(0, 0, 'b.', markersize=8)
        # make plot on top
        plt_mean_particle.set_zorder(10)
        self.plt_particles.append(plt_mean_particle)
        plt_mean_particle_ori, = plt.plot([0, 1 * np.cos(0)], [0, 1 * np.sin(0)], 'b', markersize=4)
        plt_mean_particle_ori.set_zorder(10)
        self.plt_particles_ori.append(plt_mean_particle_ori)

    
    def PlotParticles(self):
        """
        Plots all the particles and the mean particle (or best particle).
        Particles are plotted as green dots, and the mean particle is plotted as a blue dot.
        Particle orientation is plotted as a green line, and the mean particle orientation is plotted as a blue line.
        Particle size is proportional to the particle weight.
        Note that the size is scaled for visualization purposes, and does not reflect the actual weight.
        """

        # update particles
        K_size = 100 # increase the size of the particles for visualization
        K_len = 50 # increase the length of the particle vector (orientation) for visualization
        for i in range(len(self.particles)):
            self.plt_particles[i].set_data(self.particles[i][self.x_idx], self.particles[i][self.y_idx])
            self.plt_particles[i].set_markersize(self.particle_weights[i] * K_size)
            self.plt_particles_ori[i].set_data([self.particles[i][self.x_idx], self.particles[i][self.x_idx] + K_len * self.particle_weights[i] * np.cos(self.particles[i][self.yaw_idx])],
                                     [self.particles[i][self.y_idx], self.particles[i][self.y_idx] + K_len * self.particle_weights[i] * np.sin(self.particles[i][self.yaw_idx])])

        # update mean particle
        mean_particle = self.get_mean_particle()
        self.plt_particles[-1].set_data(mean_particle[self.x_idx], mean_particle[self.y_idx])
        self.plt_particles_ori[-1].set_data([mean_particle[self.x_idx], mean_particle[self.x_idx] + 1 * np.cos(mean_particle[self.yaw_idx])],
                                        [mean_particle[self.y_idx], mean_particle[self.y_idx] + 1 * np.sin(mean_particle[self.yaw_idx])])
      