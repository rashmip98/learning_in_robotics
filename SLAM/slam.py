
import os, sys, pickle, math
from copy import deepcopy

from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from load_data import load_lidar_data, load_joint_data, joint_name_to_index
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

class map_t:
    """
    This will maintain the occupancy grid and log_odds. You do not need to change anything
    in the initialization
    """
    def __init__(s, resolution=0.05):
        s.resolution = resolution
        s.xmin, s.xmax = -20, 20
        s.ymin, s.ymax = -20, 20
        s.szx = int(np.ceil((s.xmax-s.xmin)/s.resolution+1))
        s.szy = int(np.ceil((s.ymax-s.ymin)/s.resolution+1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szy), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds
        # similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received yet for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh/(1-s.occupied_prob_thresh))

    def grid_cell_from_xy(s, x, y):
        """
        x and y are 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/y go outside the map bounds, you can use
        np.clip to handle these situations.
        """

        x = np.clip(x, s.xmin, s.xmax)
        y = np.clip(y, s.ymin, s.ymax)
        x_idx = np.ceil((x + s.xmax)/s.resolution).astype(np.int16) - 1
        y_idx = np.ceil((y + s.ymax)/s.resolution).astype(np.int16) - 1
        return np.vstack((x_idx,y_idx))

class slam_t:
    """
    s is the same as self. In Python it does not really matter
    what we call self, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.05, Q=1e-3*np.eye(3),
                 resampling_threshold=0.3):
        s.init_sensor_model()

        # dynamics noise for the state (x,y,yaw)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx=0, split='train'):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar = load_lidar_data(os.path.join(src_dir,
                                               'data/%s/%s_lidar%d'%(split,split,idx)))
        s.joint = load_joint_data(os.path.join(src_dir,
                                               'data/%s/%s_joint%d'%(split,split,idx)))

        # finds the closets idx in the joint timestamp array such that the timestamp
        # at that idx is t
        s.find_joint_t_idx_from_lidar = lambda t: np.argmin(np.abs(s.joint['t']-t))

    def init_sensor_model(s):
        # lidar height from the ground in meters
        s.head_height = 0.93 + 0.33
        s.lidar_height = 0.15

        # dmin is the minimum reading of the LiDAR, dmax is the maximum reading
        s.lidar_dmin = 1e-3
        s.lidar_dmax = 30
        s.lidar_angular_resolution = 0.25
        # these are the angles of the rays of the Hokuyo
        s.lidar_angles = np.arange(-135,135+s.lidar_angular_resolution,
                                   s.lidar_angular_resolution)*np.pi/180.0

        # sensor model lidar_log_odds_occ is the value by which we would increase the log_odds
        # for occupied cells. lidar_log_odds_free is the value by which we should decrease the
        # log_odds for free cells (which are all cells that are not occupied)
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

    def init_particles(s, n=100, p=None, w=None, t0=0):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3,s.n), dtype=np.float64)
        s.w = deepcopy(w) if w is not None else np.ones(n)/float(s.n)

    @staticmethod
    def stratified_resampling(p, w):
        """
        resampling step of the particle filter, takes p = 3 x n array of
        particles with w = 1 x n array of weights and returns new particle
        locations (number of particles n remains the same) and their weights
        """

        N = p.shape[1]
        c = w[0]
        j = 0
        u = np.random.uniform(0,1.0/N)
        r_p = np.zeros(p.shape)
        r_w = np.zeros(w.shape)
        for i in range(N):
            b = u + float(i)/N
            while b > c:
                j+=1
                c+= w[j]
            r_p[:,i] = p[:,j]
            r_w[i] = 1.0/N
        return r_p, r_w


    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def rays2world(s, p, d, head_angle=0, neck_angle=0, angles=None):
        """
        p is the pose of the particle (x,y,yaw)
        angles = angle of each ray in the body frame (this will usually
        be simply s.lidar_angles for the different lidar rays)
        d = is an array that stores the distance of along the ray of the lidar, for each ray (the length of d has to be equal to that of angles, this is s.lidar[t]['scan'])
        Return an array 2 x num_rays which are the (x,y) locations of the end point of each ray
        in world coordinates
        """

        # make sure each distance >= dmin and <= dmax, otherwise something is wrong in reading
        # the data

        # 1. from lidar distances to points in the LiDAR frame

        # 2. from LiDAR frame to the body frame

        # 3. from body frame to world frame
        # if np.any(d<=s.lidar_dmin) or np.any(d>=s.lidar_dmax):
        #     raise ValueError('Corrupted lidar data')
        lidar_x = d*np.cos(angles)
        lidar_y = d*np.sin(angles)
        #T_lidar_to_body = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,s.lidar_height],[0,0,0,1]])
        T_lidar_to_body = euler_to_se3(r=0,p=head_angle, y=neck_angle, v=np.array([0,0,s.lidar_height]))
        T_body_to_world = euler_to_se3(r=0, p=0, y=p[2], v=np.array([p[0],p[1],s.head_height]))
        lidar_world_homo = T_body_to_world @ T_lidar_to_body @ np.array( [lidar_x, lidar_y, np.zeros(lidar_x.shape[0]), np.ones(lidar_x.shape[0])] )
        #lidar_world_homo = np.linalg.multi_dot([T_body_to_world, T_lidar_to_body, np.array( [lidar_x, lidar_y, np.zeros(lidar_x.shape[0]), np.ones(lidar_x.shape[0])] )])
        # print(lidar_world_homo)
        return lidar_world_homo[:,lidar_world_homo[2] > 0.0]

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d function to get the difference of the two poses and we will simply set this to be the control (delta x, delta y, delta theta)
        """

        if t == 0:
            return np.zeros(3)

        p2 = s.lidar[t]['xyth']
        p1 = s.lidar[t-1]['xyth']
        return smart_minus_2d(p2,p1)

    def dynamics_step(s, t):
        """"
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter, remember to add noise using the smart_plus_2d function to each particle
        """

        control = s.get_control(t)
        noise = np.random.multivariate_normal(np.array([0,0,0]), s.Q)
        for i in range(s.p.shape[1]):
            temp = smart_plus_2d(s.p[:,i], control)
            s.p[:,i] = smart_plus_2d(temp,noise)


    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        log_weights = np.log(w)
        log_weights += obs_logp
        log_weights = log_weights - slam_t.log_sum_exp(log_weights)
        return np.exp(log_weights)

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data

        Some notes about how to implement this.
            1. As mentioned in the writeup, for each particle
                (a) First find the head, neck angle at t (this is the same for every particle)
                (b) Project lidar scan into the world frame (different for different particles)
                (c) Calculate which cells are obstacles according to this particle for this scan,
                calculate the observation log-probability
            2. Update the particle weights using observation log-probability
            3. Find the particle with the largest weight, and use its occupied cells to update the map.log_odds and map.cells.
        You should ensure that map.cells is recalculated at each iteration (it is simply the binarized version of log_odds). map.log_odds is of course maintained across iterations.
        """

        obs_logprob = np.zeros(s.w.shape[0])
        new_t = s.find_joint_t_idx_from_lidar(s.lidar[t]['t'])
        h_angle = s.joint['head_angles'][joint_name_to_index['Head'], new_t]
        n_angle = s.joint['head_angles'][joint_name_to_index['Neck'], new_t]
        lidar_scan = s.lidar[t]['scan']
        in_range = np.logical_and(lidar_scan > s.lidar_dmin , lidar_scan <s.lidar_dmax)
        d_in_range = lidar_scan[in_range]
        angle_in_range = s.lidar_angles[in_range]
        for i in range(s.p.shape[1]):
            lidar_to_world = s.rays2world(p = s.p[:,i],d = lidar_scan, head_angle = h_angle, neck_angle = n_angle, angles=s.lidar_angles)
            map_idx = s.map.grid_cell_from_xy(lidar_to_world[0], lidar_to_world[1])
            obs_logprob[i] = np.sum(s.map.cells[map_idx[0],map_idx[1]])

        s.w = s.update_weights(s.w, obs_logprob)
        w_argmax = np.argmax(s.w)
        p_wmax = s.p[:,w_argmax]
        p_wmax_world = s.rays2world(p = p_wmax,d = d_in_range, head_angle = h_angle, neck_angle = n_angle, angles=angle_in_range)
        [start_x, start_y] = s.map.grid_cell_from_xy(p_wmax[0], p_wmax[1])
        [end_x, end_y] = s.map.grid_cell_from_xy(p_wmax_world[0], p_wmax_world[1])
        # start_x = np.tile(start_x, end_x.shape[0])
        # start_y = np.tile(start_y, end_y.shape[0])
        # idx = np.vstack((start_x, start_y, end_x, end_y))
        # for elem in idx.T:
        #     # if np.any(elem>800):
        #     #     continue
        #     s.map.log_odds[elem[2], elem[3]] += s.lidar_log_odds_occ
        #     s.map.log_odds[elem[0], elem[1]] -= s.lidar_log_odds_free
        # s.map.log_odds = np.clip(s.map.log_odds,-s.map.log_odds_max, s.map.log_odds)
        # s.map.cells = s.map.log_odds > s.map.log_odds_thresh
        elem = np.vstack((end_y, end_x))
        elem = np.hstack((elem, np.array([start_y, start_x])))
        mask = np.zeros_like(s.map.cells)

        cv2.drawContours(mask, [elem.T], contourIdx=-1, color=1, thickness=-1)

        s.map.log_odds[end_x, end_y] += (s.lidar_log_odds_occ - s.lidar_log_odds_free)
        s.map.log_odds[mask==1] += s.lidar_log_odds_free
        s.map.cells[:, :] = 0
        s.map.cells[s.map.log_odds >= s.map.log_odds_thresh] = 1
        s.map.num_obs_per_cell[mask==1] += 1
        s.resample_particles()
        # if(t%1000 == 0):
        #     MAP_2_display = 150*np.ones((s.map.cells.shape[0], s.map.cells.shape[1], 3),dtype=np.uint8)
        #     MAP_2_display = cv2.drawContours(MAP_2_display, s.map.cells, -1, (255,255,255), cv2.FILLED)
        #     #wall_indices = np.where(s.map.log_odds > s.map.log_odds_thresh)
        #     MAP_2_display = MAP_2_display - 255*s.map.cells
        #     #unexplored_indices = np.where(abs(s.map.log_odds) < 1e-1)
        #     #MAP_2_display[list(unexplored_indices[0]),list(unexplored_indices[1]),:] = [150,150,150]
        #     #MAP_2_display[start_x[0],start_y[1],:] = [0,128,0]#
        #     #ground_truth = s.lidar[t]['xyth'].reshape((3,1))
        #     #MAP_2_display[ground_truth[0], ground_truth[1]] = [128,0,0]
        #     #plt.plot(s.map.cells)
        #     plt.title('Estimated Map at time stamp %d/%d'%(t, len(s.lidar) - 0 + 1))
        #     #plt.imshow(MAP_2_display)
        #     #plt.pause(0.1)
        #     plt.imsave("map_out.png", MAP_2_display)
        #     plt.figure()

    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
