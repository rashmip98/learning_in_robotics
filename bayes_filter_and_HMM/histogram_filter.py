import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''

        N = cmap.shape[0]
        M = cmap.shape[1]
        post_dist = np.zeros((N,M))
        total = 0
        for i in range(N):
            for j in range(M):
                #action
                if i-action[1]>=0 and i-action[1]<M and j+action[0]>=0 and j+action[0]<N:
                    post_dist[i-action[1]][j+action[0]] += belief[i][j]*0.9
                    post_dist[i][j] += belief[i][j]*0.1
        
                else:
                    post_dist[i][j] += belief[i][j]*1

        for i in range(N):
            for j in range(M):
                #observation
                if cmap[i][j] == observation:
                    post_dist[i][j] = 0.9*post_dist[i][j]
                    total += post_dist[i][j]
                else:
                    post_dist[i][j] = 0.1*post_dist[i][j]
                    total += post_dist[i][j]

        post_dist = post_dist/total

        high_prob = 0

        for i in range(M):
            for j in range(N):
                if post_dist[i][j]>high_prob:
                    high_prob = post_dist[i][j]
                    pos = [i, j]

        return post_dist , pos

