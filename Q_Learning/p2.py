import gym
import numpy as np
import torch as th
import torch.nn as nn
import random

def rollout(e, q, eps=0, T=200):
    traj = []
    # Reset environment and get initial state
    x = e.reset()
    for t in range(T):
        # Get action from policy (q network)
        u = q.control(th.from_numpy(x).float().unsqueeze(0), eps=eps)
        #u = u.int().numpy().squeeze()
        # Execute action in the environment
        xp, r, d, info = e.step(u)
        t = dict(x=x, xp=xp, r=r, u=u, d=d, info=info)
        traj.append(t)
        # Update current state
        x = xp
        # If done, terminate rollout
        if d:
            break
    return traj

class q_t(nn.Module):
    def __init__(s, xdim, udim, hdim=48):
        super().__init__()
        s.xdim, s.udim = xdim, udim
        s.m = nn.Sequential(
            nn.Linear(xdim, hdim),
            nn.ReLU(True),
            nn.Linear(hdim, udim),
        )

    def forward(s, x):
        return s.m(x)

    def control(s, x, eps=0):
        # Get q values for all controls
        q = s.m(x)

        # eps-greedy strategy to choose control input
        # note that for eps=0 you should return the correct control u
        if np.random.random() < 1 - eps:
            u = q.argmax().item()
        else:
            u = np.random.randint(0, s.udim)
        return u

def loss(q, q_target, ds):

    # 1. sample mini-batch from datset ds
    # 2. code up dqn with double-q trick
    # 3. return the objective f

    batch_size = 64
    x_store = []
    xp_store = []
    r_store = []
    u_store = []
    #ds_mini_batch = np.random.shuffle(np.arange(len(ds)))[:batch_size]
    for _ in range(batch_size):
      index_1 = random.randint(0, len(ds)-1)

      index_2 = random.randint(0, len(ds[index_1]) - 1)
      elem = ds[index_1][index_2]
      if elem['d']:
          continue
      x_store.append(list(dic['x']))
      xp_store.append(list(dic['xp']))
      r_store.append(dic['r'])
      u_store.append(dic['u'])

    x = np.array(x_store)
    xp = np.array(xp_store)
    r = np.array(r_store)
    u = np.array(u_store)
    x = th.from_numpy(x).float()
    xp = th.from_numpy(xp).float()
    r = th.from_numpy(r).float().view(len(r_store), 1)
    u = th.from_numpy(u).view(len(u_store), 1)
    pred = q(x).gather(1, u)
    q_next=q(xp).detach()
    q_action=th.argmax(q_next,dim=1).reshape(-1,1)
    q_next_double=q_target(xp).detach()
    q_next_double=q_next_double.gather(1,q_action)
    target = r + 0.9 * q_next_double.max(1)[0].view(len(u_store), 1)
    f = nn.functional.mse_loss(pred, target)

    return f

def evaluate(q):

    # 1. create a new environment e
    # 2. run the learnt q network for 100 trajectories on this new environment
    # to take control actions. Remember that you should not perform
    # epsilon-greedy exploration in the evaluation phase
    # 3. report the average discounted return of these 100 trajectories
    e=gym.make('CartPole-v1')
    x=e.reset()
    eps=0
    traj=[]
    for t in range(100):
        u = q.control(th.from_numpy(x).float().unsqueeze(0),
                      eps=eps)
        xp, r, d, info = e.step(u)
        t = dict(x=x, xp=xp, r=r, u=u, d=d, info=info)
        x = xp
        traj.append(t)
        if d==True:
            break

    # rs = []
    # for elem in traj:
    #   rs.append(elem['r'])
    # R = sum([rr * 0.99 ** k for k, rr in enumerate(rs)])/len(rs)
    
    return len(traj)

if __name__=='__main__':
    # Create environment
    e = gym.make('CartPole-v1')
    xdim, udim = e.observation_space.shape[0], e.action_space.n

    # Create q network
    q = q_t(xdim, udim, 8)
    optim = th.optim.Adam(q.parameters(), lr=1e-3, weight_decay=1e-4)

    # Dataset of trajectories
    ds = []
    q_target = q_t(xdim, udim, 8)
    reward = []
    eval =[]
    loss_store = []
    # Collect few random trajectories with eps=1
    for i in range(1000):
        ds.append(rollout(e, q, eps=1, T=200))

    for i in range(1000):
        q.train()
        t = rollout(e, q)
        ds.append(t)

        # Perform weights updates on the q network
        # need to call zero grad on q function to clear the gradient buffer
        q.zero_grad()
        f = loss(q, q_target, ds)
        f.backward()
        optim.step()
        
        if i % 10 == 9:
            q_target.load_state_dict(q.state_dict())

        # Exponential averaging for the target
        reward.append(len(t))
        #print(len(t))
        loss_store.append(f.item())
        #print('Logging data to plot')
        eval.append(evaluate(q))
        #print(evaluate(q))

    #plt.plot(reward)
    #plt.plot(eval)
