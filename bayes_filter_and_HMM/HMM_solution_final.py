import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution
    
    def forward(self):

        alpha = np.zeros((self.Observations.shape[0], self.Transition.shape[0]))

        alpha[0, :] = self.Initial_distribution * self.Emission[:, self.Observations[0]]
    
        for i in range(1, self.Observations.shape[0]):
            for j in range(self.Transition.shape[0]):
                alpha[i, j] = alpha[i - 1] @ self.Transition[:, j] * self.Emission[j, self.Observations[i]]
    
        return alpha
    
    def backward(self):
    
        beta = np.zeros((self.Observations.shape[0], self.Transition.shape[0]))
 
        beta[self.Observations.shape[0] - 1] = np.ones((self.Transition.shape[0]))
    
        for i in range(self.Observations.shape[0] - 2, -1, -1):
            for j in range(self.Transition.shape[0]):
                beta[i, j] = (beta[i + 1] * self.Emission[:, self.Observations[i + 1]]) @ self.Transition[j, :]
    
        return beta
    
    def gamma_comp(self, alpha, beta):
        
        gamma = np.zeros((self.Observations.shape[0], self.Transition.shape[0]))

        for i in range(self.Observations.shape[0]):
            for j in range(self.Transition.shape[0]):
                gamma[i, j] = (alpha[i, j] * beta[i, j]) / sum(alpha[-1,:])

        return gamma
    
    def xi_comp(self, alpha, beta, gamma):
        
        xi = np.zeros((self.Observations.shape[0]-1, self.Transition.shape[0], self.Transition.shape[0]))
        temp = np.zeros((self.Transition.shape[0], self.Transition.shape[0], self.Observations.shape[0]-1))

        for i in range(self.Observations.shape[0]-1):
            denominator = (alpha[i, :].T @ self.Transition * self.Emission[:, self.Observations[i + 1]].T) @ beta[i + 1, :]
            for j in range(self.Transition.shape[0]):
                numerator = alpha[i, j] * self.Transition[j, :] * self.Emission[:, self.Observations[i + 1]].T * beta[i + 1, :].T
                temp[j, :, i] = numerator / denominator

        xi = np.transpose(temp, (2,1,0))
        return xi

    def update(self, alpha, beta, gamma, xi):

        new_init_state = np.zeros_like(self.Initial_distribution)
        T_prime = np.zeros_like(self.Transition)
        M_prime = np.zeros_like(self.Emission)
        
        new_init_state = gamma[0,:]
        gamma = np.sum(np.transpose(xi, (2,1,0)), axis=1)
        T_prime = np.sum(np.transpose(xi, (2,1,0)), 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        gamma = np.hstack((gamma, np.sum(np.transpose(xi, (2,1,0))[:, :, self.Observations.shape[0] - 2], axis=0).reshape((-1, 1))))

        K = self.Emission.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            M_prime[:, l] = np.sum(gamma[:, self.Observations == l], axis=1)

        M_prime = np.divide(M_prime, denominator.reshape((-1, 1)))

        return T_prime, M_prime, new_init_state
    
    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

        P_original = np.array([0.])
        P_prime = np.array([0.])
        hmm_prime = HMM(np.asarray(self.Observations), T_prime, M_prime, new_init_state)
        alpha_prime = hmm_prime.forward()
        P_original = sum(alpha[-1,:])
        P_prime = sum(alpha_prime[-1,:])
        return P_original, P_prime


