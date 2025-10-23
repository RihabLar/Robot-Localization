# Particle Filter for Robot Localization

This project implements a **Particle Filter** (Monte Carlo Localization) for a differential drive mobile robot.  
The particle filter is a Bayesian approach that represents the robot’s belief of its state as a set of weighted particles, updated through motion and sensor models.

## Overview
- **Prediction step**: Propagates particles using odometry data with added noise to model uncertainty.  
- **Update step**: Adjusts particle weights based on sensor measurements and likelihood functions.  
- **Resampling**: Implements both Roulette Wheel and Stochastic Universal Sampling to maintain particle diversity.  
- **Simulation**: Demonstrates how particles converge towards the robot’s true pose over time, even under noisy motion and sensor data.

## Results
- Particles spread during prediction, reflecting motion uncertainty.  
- With sensor updates, particles converge towards the true robot pose.  
- Resampling ensures diversity and prevents particle depletion.  
- Outperforms dead reckoning by preventing error accumulation.  


