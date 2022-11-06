import argparse
import numpy as np
from scipy import io
import math
import matplotlib.pyplot as plt

#this function takes in an input number (1 through 3)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    #imu = io.loadmat('source/imu/imuRaw'+str(data_num)+'.mat')
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]

    vicon_rot = vicon['rots']
    vicon_t = vicon['ts'].T
    true_roll = np.zeros((vicon_rot.shape[2],))
    true_pitch = np.zeros((vicon_rot.shape[2],))
    true_yaw = np.zeros((vicon_rot.shape[2],))
    for i in range(vicon_rot.shape[2]):
       true_roll[i], true_pitch[i], true_yaw[i] = euler_angles(from_rotm(vicon_rot[:,:,i]))

    # Multiplying ax and ay with -1
    accel[0:2,:] = accel[0:2,:]*-1
    accel = accel.T
    #print(accel)

    # Rearranging gyro such that it is gx, gy, gz
    gyro = gyro[[1,2,0],:]
    #print(gyro.shape)
    gyro = gyro.T


    # Sensitivity and bias for accelerometer, gyro
    vref = 3300
    accel_alpha = 332.17 #32*9.81
    accel_bias =   [65025, 65035, 503] #[65025, 65035, 503] #np.mean(accel[:10], axis = 0) - np.array([0,0,1])/(vref/(1023*accel_alpha)) #[511, 501, 503]
    gyro_alpha = 193.55 #195
    gyro_bias =  [374.5, 376, 370] #[374.5, 376, 370] #np.mean(gyro[:10], axis = 0) #[369.5, 371.5, 377]

    accel = (accel - accel_bias)*vref/(1023*accel_alpha)
    gyro = (gyro - gyro_bias)*vref/(1023*gyro_alpha)

    #Initialize the covariances 
    P_quat = 0.1*np.identity(3)
    P_omega = 0.1*np.identity(3)
    # Process noise covariance 
    Q_quat = 2*np.identity(3)
    Q_omega = 2*np.identity(3)
    # Measurement noise covariance 
    R_quat = 2*np.identity(3)
    R_omega = 2*np.identity(3)

    q_t = np.array([1,0,0,0])
    q_est = q_t
    omega_t = np.array([0.1,0.1,0.1])
    omega_est = omega_t
    q_variance = np.array([[0.1,0.1,0.1]])
    omega_variance = np.array([[0.1,0.1,0.1]])

    for t in range(T):
        # Calculating Xi 
        #print('Start')
        P_rows, P_cols = P_quat.shape # 3x3
        S_quat = np.linalg.cholesky(P_quat+Q_quat) # 3x3
        Wi_left_quat = S_quat * np.sqrt(2*P_rows) #3x3
        Wi_right_quat = -S_quat * np.sqrt(2*P_rows) # 3x3
        Wi_quat = np.hstack((Wi_left_quat, Wi_right_quat)) # 3x6
        
        Wi_quat = Wi_quat.T #12x6
        X_i_quat = np.zeros((2*P_rows, 4)) # 12x7

        S = np.linalg.cholesky(P_omega+Q_omega) # 6x6
        Wi_left = S * np.sqrt(2*P_rows) #6x6
        Wi_right = -S * np.sqrt(2*P_rows) # 6x6
        Wi_omega = np.hstack((Wi_left, Wi_right)) # 6x12
        
        Wi_omega = Wi_omega.T #12x6
        X_i_omega = np.zeros((2*P_rows, 3)) # 12x7

        for i in range(2*P_rows): 
            Wi_temp = from_axis_angle(Wi_quat[i,:]) 
            X_i_quat[i, :] = multiply(q_t, Wi_temp) 
            X_i_omega[i,:] = omega_t + Wi_omega[i,:]   
        # if(t==1):
        #     print(Wi_left)
        #     print(Wi_right)
        #print(imu['ts'].shape)
        
        # Calculating Yi
        if t != T-1:
            delta_t = imu['ts'].T[i+1] - imu['ts'].T[i]
        else:
            delta_t = imu['ts'].T[-1] - imu['ts'].T[-2]

        Xi_rows = X_i_quat.shape[0] # 12
        Y_i_quat = np.zeros((Xi_rows,4)) # 12x7
        q_delta = from_axis_angle(gyro[t,:]*delta_t) 
        for i in range(Xi_rows):
            q = X_i_quat[i] # 4
            Y_i_quat[i] = multiply(q, q_delta) 
        
        # Now for xkbar and Pkbar
        x_k_bar_omega = np.mean(X_i_omega, axis=0) # 3x1
        x_k_bar_quat, Wi_prime_quat = quat_mean(Y_i_quat, q_t) # 4x1
 
        Wi_prime_omega =  X_i_omega - x_k_bar_omega

        P_k_bar_quat = np.zeros((3,3))
        P_k_bar_omega = np.zeros((3,3))

        P_k_bar_quat = (Wi_prime_quat.T@Wi_prime_quat)/6
        P_k_bar_omega = (Wi_prime_omega.T@Wi_prime_omega)/6

        # Now for measurement model
        g = np.array([0, 0, 0, 1])
        Z_i_quat = np.zeros((6, 3)) # vector quaternions
        for i in range(6):
            q = Y_i_quat[i]
            Z_i_quat[i] = multiply(multiply(inv(q), g), q)[1:] 
   
        z_k_bar_quat = np.mean(Z_i_quat, axis=0) # 6
        z_k_bar_quat /= np.linalg.norm(z_k_bar_quat)

        z_k_bar_omega = np.mean(X_i_omega, axis=0) # 6
        z_k_bar_omega /= np.linalg.norm(z_k_bar_omega)
        

        P_zz_quat = np.zeros((3,3))
        P_xz_quat = np.zeros((3,3))
        Wi_z_quat = Z_i_quat - z_k_bar_quat

        P_zz_omega = np.zeros((3,3))
        P_xz_omega = np.zeros((3,3))
        Wi_z_omega = X_i_omega - z_k_bar_omega 
        
        P_zz_quat =(Wi_z_quat.T@Wi_z_quat)/ 6
        P_xz_quat = (Wi_prime_quat.T@Wi_z_quat)/ 6

        P_zz_omega =(Wi_z_omega.T@Wi_z_omega)/ 6
        P_xz_omega = (Wi_prime_omega.T@Wi_z_omega)/ 6
        if t==0:
            print(P_xz_omega)
        
        acc = accel[t]/np.linalg.norm(accel[t])
        v_k_quat = acc - z_k_bar_quat  # 6
        v_k_omega = gyro[t] - z_k_bar_omega
        
        P_vv_quat = P_zz_quat + R_quat # 6x6
        P_vv_omega = P_zz_omega + R_omega
        K_quat = np.dot(P_xz_quat,np.linalg.inv(P_vv_quat)) #6x6
        K_omega = np.dot(P_xz_omega,np.linalg.inv(P_vv_omega))
        update_quat = K_quat.dot(v_k_quat) # 6
        update_omega = K_omega.dot(v_k_omega)

        q_pred = multiply(from_axis_angle(update_quat), x_k_bar_quat)
        omega_pred = x_k_bar_omega + update_omega
        P_quat = P_k_bar_quat - K_quat.dot(P_vv_quat).dot(K_quat.T) 
        P_omega = P_k_bar_omega - K_omega.dot(P_vv_omega).dot(K_omega.T) 
        q_est = np.vstack((q_est, q_pred))
        omega_est = np.vstack((omega_est, omega_pred))
        q_t = q_pred
        omega_t = omega_pred
        q_variance = np.vstack((q_variance, np.array([[P_quat[0][0], P_quat[1][1], P_quat[2,2]]])))
        omega_variance = np.vstack((omega_variance, np.array([[P_omega[0][0], P_omega[1][1], P_omega[2][2]]])))

    roll = np.zeros(np.shape(q_est)[0])
    pitch = np.zeros(np.shape(q_est)[0])
    yaw = np.zeros(np.shape(q_est)[0])

    for i in range(np.shape(q_est)[0]):
        roll[i], pitch[i], yaw[i] = euler_angles(q_est[i])
    
    ## Code to plot the true and UKF estimmates
    fig  = plt.figure()
    plt.subplot(3,1,1)
    plt.plot(roll, label = 'UKF')
    plt.plot(true_roll, label = 'True')
    plt.legend(loc='best')
    plt.subplot(3,1,2)
    plt.plot(pitch, label = 'UKF')
    plt.plot(true_pitch, label = 'True')
    plt.legend(loc='best')
    plt.subplot(3,1,3)
    plt.plot(yaw, label = 'UKF')
    plt.plot(true_yaw, label = 'True')
    plt.legend(loc='best')
    gx = np.zeros(np.shape(omega_est)[0])
    gy = np.zeros(np.shape(omega_est)[0])
    gz = np.zeros(np.shape(omega_est)[0])
    for i in range(np.shape(omega_est)[0]):
        gx[i], gy[i], gz[i] = omega_est[i]
    plt.subplot(3,1,1)
    plt.plot(gx, label = 'UKF: omega_x')
    plt.legend(loc='best')
    plt.subplot(3,1,2)
    plt.plot(gy, label = 'UKF: omega_y')
    plt.legend(loc='best')
    plt.subplot(3,1,3)
    plt.plot(gz, label = 'UKF: omega_z')
    plt.legend(loc='best')
    plt.plot(q_variance[:,0], label = 'Quaternion Variance 1')
    plt.plot(q_variance[:,1], label = 'Quaternion Variance 2')
    plt.plot(q_variance[:,2], label = 'Quaternion Variance 3')
    plt.legend(loc='best')
    plt.plot(omega_variance[:,0], label = 'Omega Variance 1')
    plt.plot(omega_variance[:,1], label = 'Omega Variance 2')
    plt.plot(omega_variance[:,2], label = 'Omega Variance 3')
    plt.legend(loc='best')
    plt.show()



    # roll, pitch, yaw are numpy arrays of length T
    #print(roll.shape, T)
    
    return roll,pitch,yaw


# Helper functions for quaternions
def from_rotm(R):
        q = np.zeros(4)
        theta = math.acos((np.trace(R)-1)/2)
        omega_hat = (R - np.transpose(R))/(2*math.sin(theta))
        omega = np.array([omega_hat[2,1], -omega_hat[2,0], omega_hat[1,0]])
        q[0] = math.cos(theta/2)
        q[1:4] = omega*math.sin(theta/2)
        normalize(q)
        return q

def axis_angle(q):
        scalar = q[0]
        vec = q[1:4]
        theta = 2*math.acos(scalar) #changed scalar
        if (np.linalg.norm(vec) == 0):
            return np.zeros(3)
        vec = vec/np.linalg.norm(vec)
        return vec*theta

def from_axis_angle(a):
        q = np.zeros(4)
        angle = np.linalg.norm(a)
        if angle != 0:
            axis = a/angle
        else:
            axis = np.array([1,0,0])
        q[0] = math.cos(angle/2)
        q[1:4] = axis*math.sin(angle/2)
        return q

def euler_angles(q):
        phi = math.atan2(2*(q[0]*q[1]+q[2]*q[3]), \
                1 - 2*(q[1]**2 + q[2]**2))
        theta = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
        psi = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), \
                1 - 2*(q[2]**2 + q[3]**2))
        return phi, theta, psi

def normalize(q):
        q = q/np.linalg.norm(q)
        return q

def inv(q):
        q_inv = np.array([q[0], -1*q[1], -1*q[2], -1*q[3]])
        q_inv = normalize(q_inv)
        return q_inv

def multiply(q, other):
        res = np.zeros(4)
        res[0] = q[0]*other[0] -1*q[1]*other[1] -1*q[2]*other[2] -1*q[3]*other[3]
        res[1] = q[0]*other[1] + q[1]*other[0] + q[2]*other[3] -1*q[3]*other[2]
        res[2] = q[0]*other[2] -1*q[1]*other[3] + q[2]*other[0] + q[3]*other[1]
        res[3] = q[0]*other[3] + q[1]*other[2] -1*q[2]*other[1] + q[3]*other[0]
        return res

def quat_mean(q_1, q_2): 
    q_t = q_2

    n = q_1.shape[0] 

    eps = 0.0001

    e = np.zeros((n,3)) 
    for _ in range(1000):
        for i in range(n):
            q_e = normalize(multiply(q_1[i, :], inv(q_t))) 
            e_e = axis_angle(q_e) 
            if np.linalg.norm(e_e) == 0: 
                e[i:] = np.zeros(3)
            else:
                e[i,:] = (-np.pi + np.mod(np.linalg.norm(e_e) + np.pi, 2 * np.pi)) / np.linalg.norm(e_e) * e_e
        e_mean = np.mean(e, axis=0)
        q_t = normalize(multiply(from_axis_angle(e_mean), q_t))
        if np.linalg.norm(e_mean) < eps:
            return q_t, e
        e = np.zeros((n,3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=int, default=3)
    args = parser.parse_args()

    r, p, y = estimate_rot(args.data)