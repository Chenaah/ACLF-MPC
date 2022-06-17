from cProfile import label
from cmath import sin
import math
from re import A
from turtle import color, pd
import numpy as np
import matplotlib.pyplot as plt
from pyro import param
from scipy.integrate import odeint
import pdb
from torch import zero_
# from cem import CEM
# from cmaes import CMA
from tqdm import tqdm
import nevergrad as ng
# from cem_pro import CEMOptimizer
from cbo import optimise
import seaborn as sns

EPISODE_LEN = 40 #s
HORIZON_LEN = 5 #s

class Plant(object):
    def __init__(self) -> None:
        self.time_step = 0.01
        self.state = np.array([0, 0, 0, 0])
        self.state_dot = np.array([0, 0, 0, 0])
        self.state_dotdot = np.array([0, 0, 0, 0])
        self.t = 0
        self.time = np.arange(0, EPISODE_LEN, self.time_step)

        self.ref_x = 0.8

        self.m1, self.m2, self.k1, self.k2, self.b1, self.b2 = 1.2, 1.2, 0.9, 0.4, 0.8, 0.4
        self.param_star = [self.m1, self.m2, self.k1, self.k2, self.b1, self.b2]

    def reset(self):
        self.state = np.array([0, 0, 0, 0])
        self.last_state = np.array([0, 0, 0, 0])
        self.last_state_dot = np.array([0, 0, 0, 0])
        self.t = 0
        return self.state

    def model(self, a, t, f):
        
        return np.array([a[2],
                        a[3],
                        (-(self.k1 + self.k2) * a[0] - self.b1 * a[2] + f + self.k2 * a[1]) * (1 / self.m1),
                        (-self.k2 * a[1] + self.k2 * a[0] - self.b2 * a[3]) * (1 / self.m2)])

    def __call__(self, a, t, f):
        return self.model(a, t, f)

    def step(self, action):
        assert type(action) == float
        y0 = self.state
        try:
            traj = odeint(self.model, y0, self.time[:2], args=(action,))
        except TypeError:
            pdb.set_trace()
        self.state = traj[1]
        self.state_dot = (self.state - self.last_state) / self.time_step
        self.state_dotdot = (self.state_dot - self.last_state_dot) / self.time_step

        

        self.last_state = np.array([i for i in self.state])
        self.last_state_dot = np.array([i for i in self.state_dot])



        return self.state, self.state_dot, self.state_dotdot

class MPC(object):
    def __init__(self) -> None:
        self.time_step = 0.01
        self.state = np.array([0, 0, 0, 0])
        self.t = 0
        self.time = np.arange(0, EPISODE_LEN, self.time_step)
        self.horizon_step = int(HORIZON_LEN/self.time_step)
        ref_x = 0.8
        self.ref = np.array([[ref_x,ref_x,ref_x,ref_x]]*(self.horizon_step+len(self.time)))
        # for i in range(self.horizon_step+len(self.time)):
        #     self.ref[i][0] = 0.25*math.sin(0.01*i)
        #     # self.ref[i][1] = 0.25*math.sin(0.01*i)
        #     self.ref[i][1] = 0.025*i
        #     # print(math.sin(0.01*i)+1)

        # pdb.set_trace()
        self.pi_u = np.zeros((6,1))
        self.tau_u = np.zeros((2,1))

        delta_m = 0.5
        self.m1, self.m2, self.k1, self.k2, self.b1, self.b2 = 1.2+0.6, 1.2+0.6, 0.9+0.45, 0.4+0.2, 0.8+0.4, 0.4+0.2
        self.pi_n = [[self.m1], [self.k1+self.k2], [self.k2], [self.b1], [self.m1/self.m2*self.k2], [self.m1/self.m2*self.b2]] # 6x1

        self.m1_u, self.m2_u, self.k1_u, self.k2_u, self.b1_u, self.b2_u = 0, 0, 0, 0, 0, 0
        self.m1_hat, self.m2_hat, self.k1_hat, self.k2_hat, self.b1_hat, self.b2_hat = self.m1, self.m2, self.k1, self.k2, self.b1, self.b2

        self.param_hat = [self.m1_hat, self.m2_hat, self.k1_hat, self.k2_hat, self.b1_hat, self.b2_hat]

        self.tau_n = np.zeros((2,1))
        self.Y = np.zeros((2,6))


    def track_params(self):
        self.m1_u = self.pi_u[0]
        self.m2_u = self.pi_u[0]*self.pi_u[2]/self.pi_u[4]
        self.k1_u = self.pi_u[1] - self.pi_u[2]
        self.k2_u = self.pi_u[2]
        self.b1_u = self.pi_u[3]
        self.b2_u  = self.pi_u[0]*self.pi_u[2]*self.pi_u[5]/self.pi_u[4]

        # self.m1_hat = self.m1 + self.m1_u
        # self.m2_hat = self.m2 + self.m2_u
        # self.k1_hat = self.k1 + self.k1_u
        # self.k2_hat = self.k2 + self.k2_u
        # self.b1_hat = self.b1 + self.b1_u
        # self.b2_hat = self.b2 + self.b2_u

        self.m1_hat = self.pi_u[0] + self.pi_n[0]
        self.m2_hat = (self.pi_u[0]+self.pi_n[0])*(self.pi_u[2]+self.pi_n[2])/(self.pi_u[4]+self.pi_n[4])
        self.k1_hat = (self.pi_u[1]+self.pi_n[1]) - (self.pi_u[2]+self.pi_n[2])
        self.k2_hat = self.pi_u[2]+self.pi_n[2]
        self.b1_hat = self.pi_u[3]+self.pi_n[3]
        self.b2_hat = (self.pi_u[0]+self.pi_n[0])*(self.pi_u[2]+self.pi_n[2])*(self.pi_u[5]+self.pi_n[5])/(self.pi_u[4]+self.pi_n[4])


        self.param_hat = [self.m1_hat, self.m2_hat, self.k1_hat, self.k2_hat, self.b1_hat, self.b2_hat]





    def update_model(self, pi_dot, Y):
        self.Y = Y
        self.pi_u += pi_dot # 6x1
        self.tau_u = np.matmul(Y, self.pi_u) # 2x1
        self.tau_n = np.matmul(Y, self.pi_n)
        assert np.shape(self.tau_u) == (2,1)

    def model(self, a, t, f):

        m1, m2, k1, k2, b1, b2 = self.m1, self.m2, self.k1, self.k2, self.b1, self.b2 
        # pdb.set_trace()
        return np.array([a[2],
                        a[3],
                        (-(k1 + k2) * a[0] - b1 * a[2] + f + self.tau_u[0].item() + k2 * a[1]) * (1 / m1),
                        (-k2 * a[1] + k2 * a[0] - b2 * a[3]) * (1 / m2)])


    def traj_reward_const_u(self, s0, a_, t_curr):
        assert type(a_) == float
        horizon = np.arange(0, HORIZON_LEN, self.time_step)
        # pdb.set_trace()
        s_horizon = odeint(self.model, s0, self.time[:self.horizon_step], args=(a_,))

        s_horizon = np.array(s_horizon[:len(horizon)])
        # pdb.set_trace()
        try:
            # error = np.mean(np.square(s_horizon[:,:2] - self.ref[t_curr:t_curr+len(horizon),:2]))
            error = np.mean(np.square(s_horizon[:,1] - self.ref[t_curr:t_curr+len(horizon),1]))
        except ValueError:
            pdb.set_trace()
        return -error


    def get_action(self, state, t_curr, Y, sigma):

        def cost(a):
            assert type(a) == float
            # print(self.barrier(h_clf(a)))
            gamma_param = 0.05
            return -self.traj_reward_const_u(state, a, t_curr) + self.barrier(h_clf(a)) + gamma_param*np.linalg.norm(np.array([[a],[0]]) - np.matmul(Y, self.pi_u) - np.matmul(Y, self.pi_n))

        def h_clf(u):
            # assert np.shape(sigma) == (2,)
            K_d = np.full((2, 2), 1)
            # pdb.set_trace()
            # return (-np.matmul(sigma, (np.array([np.matmul(self.pi_n, Y)]).T + np.array([np.matmul(self.pi_u, Y)]).T - np.array([[tau], [0]]))) - 0.5*np.matmul(np.matmul(sigma, K_d), sigma.T)).item()
            return (-np.matmul(sigma.T, (   np.matmul(Y, np.array(self.pi_n)) + np.matmul(Y, self.pi_u) - np.array([[u], [0]])   )) - 0.5*np.matmul(np.matmul(sigma.T, K_d), sigma)).item()

        optimizer = ng.optimizers.NGOpt(parametrization=ng.p.Scalar(init=0, lower=0, upper=2), budget=50, num_workers=1) #GOOD
        # optimizer.parametrization.register_cheap_constraint(lambda x: h_clf(x) >= 0)

        # print(optimizer._select_optimizer_cls(), "IS USED FOR OPTIMISATION!")
        a_star = optimizer.minimize(cost).value  # best value
        # best_y, A = optimise(reward, const, 1)
        # a_star = float(A[0])
        self.curr_cost = -self.traj_reward_const_u(state, a_star, t_curr)

        if type(a_star) != float:
            pdb.set_trace()

        return a_star

    def barrier(self, h, delta=50):
        if h < delta:
            return 0.5*(((h-2*delta)/delta)**2-1) - math.log(delta)
        else:
            return -math.log(h)



if __name__ == '__main__':

    plant = Plant()
    s = plant.reset()
    agent = MPC()
    h = 0.01

    # simulation time
    time = np.arange(0, EPISODE_LEN, 0.01)
    horizon = np.arange(0, 40, h)

    # initial conditions

    # solve system of equations args=(m1, m2, k1, k2, b1, b2)

    ref_x = 0.8

    s_list = []
    a_list = []
    a_n_list = []
    a_u_list = []
    costs = []
    sigmas = []
    Y = np.zeros((2,6))
    sigma = np.zeros(2)

    param_list = []

    for t in tqdm(range(len(time))):

        action = agent.get_action(s, t, Y, sigma)
        # print("ACTION: ", action)
        s_, s_d, s_dd = plant.step(action)

        costs.append(agent.curr_cost)

        lamda = 1
        # pdb.set_trace()
        sigma = np.array([[(s_d[0] - 0) + lamda*(s_[0] - ref_x), (s_d[1] - 0) + lamda*(s_[1] - ref_x)]]).T # 2x1
        sigmas.append(sigma.T.flatten())
        Y = np.array([[s_dd[0], s_[0], -s_[1], s_d[0], 0, 0], 
                      [s_dd[1], 0, 0, 0, -s_[0]+s_[1], s_d[1]]]) # 2x6
        gamma = 0.002
        # pdb.set_trace()
        pi_d = gamma*np.matmul(Y.T, sigma) # 6x1
        # pdb.set_trace()

        param_list.append(agent.param_hat)

        agent.update_model(pi_d, Y)

        agent.track_params()

        s_list.append(s_)
        a_list.append(action)
        a_n_list.append(agent.tau_n[0])
        a_u_list.append(agent.tau_u[0])

    s_list = np.array(s_list)
    param_list = np.array(param_list)

    plant = Plant()
    s = plant.reset()
    agent = MPC()


    s_list_nadapt = []
    a_list_nadapt = []
    Y = np.zeros((2,6))
    sigma = np.zeros(2)

    for t in tqdm(range(len(time))):

        action = agent.get_action(s, t, Y, sigma)
        # print("ACTION: ", action)
        s_, s_d, s_dd = plant.step(action)

        lamda = 1
        # pdb.set_trace()
        sigma = np.array([[(s_d[0] - 0) + lamda*(s_[0] - ref_x), (s_d[1] - 0) + lamda*(s_[1] - ref_x)]]).T # 2x1
        Y = np.array([[s_dd[0], s_[0], -s_[1], s_d[0], 0, 0], 
                      [s_dd[1], 0, 0, 0, -s_[0]+s_[1], s_d[1]]]) # 2x6
        gamma = 0.002
        # pdb.set_trace()
        pi_d = gamma*np.matmul(Y.T, sigma) # 6x1
        # pdb.set_trace()
        # agent.update_model(pi_d, Y)

        s_list_nadapt.append(s_)
        a_list_nadapt.append(action)

    # pdb.set_trace()
    sigmas = np.array(sigmas)
    s_list_nadapt = np.array(s_list_nadapt)
    # s_list_nadapt = np.array(s_list_nadapt)

    sns.set_theme()

    fig, ax = plt.subplots(figsize=(10,5))
    cmap = plt.get_cmap("tab10")
    ax.plot(time, s_list[:, 0], color=cmap(0), label='$m_1$ Displacement')
    ax.plot(time, s_list[:, 1], color=cmap(1), label='$m_2$ Displacement')
    # ax.plot(time, a_list, color=cmap(2), label="Input $F$")
    ax.plot(time, s_list_nadapt[:, 0], color=cmap(0), label='$m_1$ Displacement (without adaptation)', linestyle='--')
    ax.plot(time, s_list_nadapt[:, 1], color=cmap(1), label='$m_2$ Displacement (without adaptation)', linestyle='--')
    ax.plot(time, agent.ref[:len(time), 0], linestyle='--', color=cmap(3), label='Reference')
    # ax.plot(time, agent.ref[:len(time), 1], linestyle='--', color=cmap(1), label='m2 Reference')

    ax.set(xlabel='time (s)', ylabel='x1, x2 (m)', title='States of the System')
    ax.legend(loc='lower right')
    # ax.grid()
    plt.savefig("pro.pdf")

    fig, ax = plt.subplots(figsize=(6,4))
    for i, lab, true_value in zip(range(6), ['m_1', 'm_2', 'k_1', 'k_2', 'b_1', 'b_2'], plant.param_star):
        ax.plot(time, param_list[:, i], color=cmap(i), label=f"${lab}$")
        ax.plot(time, [true_value]*len(time), color=cmap(i), label=f"${lab}^*$", linestyle='--')
    ax.set(xlabel='time (s)', ylabel='Value', title='Parameters of the System')
    ax.legend(loc='lower right')
    # ax.grid()
    plt.savefig("param.pdf")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(time, a_list, color=cmap(0), label="Adaptive Input $F$")
    ax.plot(time, a_n_list, color=cmap(1), label="$\\tau_n$")
    ax.plot(time, a_u_list, color=cmap(2), label="$Y\\hat{\\pi}_u$")
    ax.set(xlabel='time (s)', ylabel='Force', title='Adaptive Force')
    ax.legend(loc='lower right')
    # ax.grid()
    plt.savefig("force.pdf")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(time, sigmas[:,0], label="Composite Error $\\sigma_0$")
    ax.plot(time, sigmas[:,1], label="Composite Error $\\sigma_1$")
    ax.set(xlabel='time (s)', ylabel='Root Mean Square Error (RMSE)', title='Adaptive Force')
    ax.legend(loc='lower right')
    # ax.grid()

    plt.savefig("rmse.pdf")

    np.save("data/s_list.npy", s_list)
    np.save("data/s_list_nadapt.npy", s_list_nadapt)
    np.save("data/ref.npy", agent.ref[:len(time), 0])
    np.save("data/param_list.npy", param_list)
    np.save("data/param_star.npy", plant.param_star)
    np.save("data/a_list.npy", a_list)
    np.save("data/a_n_list.npy", a_n_list)
    np.save("data/a_u_list.npy", a_u_list)
    np.save("data/costs.npy", costs)
    np.save("data/sigmas.npy", sigmas)


    plt.show()