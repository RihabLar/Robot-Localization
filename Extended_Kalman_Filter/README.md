# Extended Kalman Filter (EKF) for Robot Localization

This project implements the **Extended Kalman Filter (EKF)** for a 3DOF differential drive mobile robot.  
The EKF is a Gaussian-based filter that extends the Kalman Filter to handle **nonlinear motion and observation models** by linearizing them around the current estimate.

## Overview
- **Dead Reckoning + Compass EKF**: Combines odometry displacement with compass updates to refine orientation estimates.  
- **Constant Velocity Model**: Extended the state vector to include velocities and yaw rate for more realistic motion modeling.  
- **Map-Based EKF Localization**: Incorporated known map features into the observation model, using data association (Mahalanobis distance + ICNN) to match observed features with expected ones.  
- **Cartesian & Polar Features**: Implemented transformations and Jacobians for handling both Cartesian and Polar feature representations.  

## Results
- Without compass updates, localization drifted significantly due to uncorrected orientation error.  
- Frequent compass updates improved accuracy and reduced uncertainty.  
- Map-based EKF with feature updates every 50 steps produced accurate localization and reduced uncertainty ellipses.  
- Demonstrated robustness of EKF in handling nonlinearities and noisy sensor data.  


