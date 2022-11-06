## Description
This implementation of the UKF follows the algorithm described in the paper [here](https://github.com/rashmip98/learning_in_robotics/blob/main/unscented_kalman_filter/ukf_writeup.pdf)

## Usage
* There are 3 data files each in the vicon and imu folders. To switch between using the different data files, type either 1, 2 or 3 when running the code for the `data` argument.
 
To run the code:
```
python estimate_rot.py --data 1/2/3
```

A plot of the true data and the estimated UKF values can be found [here](https://github.com/rashmip98/learning_in_robotics/blob/main/unscented_kalman_filter/quat_mean.png)
