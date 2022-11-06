import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']

    print("belief_states: \n", belief_states)
    #print(belief_states.shape)
    #print(cmap.shape)
    #print(actions.shape)
    #print(actions)
    #out = np.zeros((cmap.shape[0], cmap.shape[1]))
    #print(out[0])
    #print(observations)


    #### Test code here
    N = cmap.shape[0]
    M = cmap.shape[1]
    prior_belief = np.full((N,M), 1/(N*M))
    bayes_filter = HistogramFilter()
    state = []
    for i in range(len(actions)):
        belief_new, pos = bayes_filter.histogram_filter(cmap, prior_belief, actions[i], observations[i])
        state.append(pos)
    #print("new belief_states: \n", belief_new)
    print("state: ", state)
