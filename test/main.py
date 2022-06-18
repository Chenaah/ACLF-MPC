from cProfile import label
from cmath import sin
import math
from re import A
from turtle import color, pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pdb
# from cem import CEM
# from cmaes import CMA
from tqdm import tqdm
import nevergrad as ng
# from cem_pro import CEMOptimizer
from cbo import optimise

EPISODE_LEN = 30 #s
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

    def reset(self):
        self.state = np.array([0, 0, 0, 0])
        self.last_state = np.array([0, 0, 0, 0])
        self.last_state_dot = np.array([0, 0, 0, 0])
        self.t = 0
        return self.state

    def model(self, a, t, f):
        m1, m2, k1, k2, b1, b2 = 1.2, 1.2, 0.9, 0.4, 0.8, 0.4
        return np.array([a[2],
                        a[3],
                        (-(k1 + k2) * a[0] - b1 * a[2] + f + k2 * a[1]) * (1 / m1),
                        (-k2 * a[1] + k2 * a[0] - b2 * a[3]) * (1 / m2)])

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
        self.pi_u = np.zeros((2,6))
        self.tau_u = np.zeros(2)

        delta_m = 0.5
        self.m1, self.m2, self.k1, self.k2, self.b1, self.b2 = 1.2+0.6, 1.2+0.6, 0.9+0.45, 0.4+0.2, 0.8+0.8, 0.4+0.2
        self.pi_n = [[self.m1, 0, self.b1, 0, self.k1+self.k2, -self.k2], [0, self.m2, 0, self.b2, -self.k2, self.k2]]

    # def pi

    def update_model(self, pi_dot, Y):
        self.pi_u += pi_dot.T
        self.tau_u = np.matmul(Y, self.pi_u.T)
        pdb.set_trace()
        assert np.shape(self.tau_u) == (2,)

    def model(self, a, t, f):

        m1, m2, k1, k2, b1, b2 = self.m1, self.m2, self.k1, self.k2, self.b1, self.b2 
        return np.array([a[2],
                        a[3],
                        (-(k1 + k2) * a[0] - b1 * a[2] + f + self.tau_u[0] + k2 * a[1]) * (1 / m1),
                        (-k2 * a[1] + k2 * a[0] - b2 * a[3]) * (1 / m2)])

    def traj_reward(self, s0, A, t_curr):
        horizon = np.arange(0, HORIZON_LEN, self.time_step)
        s_ = s0
        s_horizon = [s_]
        for a_ in A:
            traj = odeint(self.model, s_, self.time[:2], args=(a_,))
            s_ = traj[1]
            s_horizon.append(s_)
        s_horizon = np.array(s_horizon[:len(horizon)])
        # pdb.set_trace()
        try:
            error = np.mean(np.square(s_horizon[:,:2] - self.ref[t_curr:t_curr+len(horizon),:2]))
            # error = np.mean(np.square(s_horizon[:,1] - self.ref[t_curr:t_curr+len(horizon),1]))
        except ValueError:
            pdb.set_trace()
        return -error

    def traj_reward_const_u(self, s0, a_, t_curr):
        horizon = np.arange(0, HORIZON_LEN, self.time_step)
        pdb.set_trace()
        s_horizon = odeint(self.model, s0, self.time[:self.horizon_step], args=(a_,))

        s_horizon = np.array(s_horizon[:len(horizon)])
        # pdb.set_trace()
        try:
            # error = np.mean(np.square(s_horizon[:,:2] - self.ref[t_curr:t_curr+len(horizon),:2]))
            error = np.mean(np.square(s_horizon[:,1] - self.ref[t_curr:t_curr+len(horizon),1]))
        except ValueError:
            pdb.set_trace()
        return -error

    def test(self):
        print(self.traj_reward([0,0,0,0], [4]*self.horizon_step, 0))

    def get_action(self, state, t_curr, Y, sigma):

        def cost(a):
            # print(self.barrier(h_clf(a)))
            return -self.traj_reward_const_u(state, a, t_curr) + self.barrier(h_clf(a))

        def reward(a):
            # pdb.set_trace()
            # print(self.barrier(h_clf(a)))
            return self.traj_reward_const_u(state, a[0], t_curr)

        def const(a):
            return -h_clf(a[0])

        def h_clf(tau):
            # assert np.shape(sigma) == (2,)
            K_d = np.full((2, 2), 1)
            # pdb.set_trace()
            # return (-np.matmul(sigma, (np.array([np.matmul(self.pi_n, Y)]).T + np.array([np.matmul(self.pi_u, Y)]).T - np.array([[tau], [0]]))) - 0.5*np.matmul(np.matmul(sigma, K_d), sigma.T)).item()
            return (-np.matmul(sigma, (np.array([np.matmul(Y, np.array(self.pi_n).T)]).T + np.array([np.matmul(Y, self.pi_u.T)]).T - np.array([[tau], [0]]))) - 0.5*np.matmul(np.matmul(sigma, K_d), sigma.T)).item()

        optimizer = ng.optimizers.NGOpt(parametrization=ng.p.Scalar(init=0, lower=-1, upper=1), budget=50, num_workers=1) #GOOD
        # print(optimizer._select_optimizer_cls(), "IS USED FOR OPTIMISATION!")
        a_star = optimizer.minimize(cost).value  # best value
        # best_y, A = optimise(reward, const, 1)
        # a_star = float(A[0])

        if type(a_star) != float:
            pdb.set_trace()

        return a_star

    def barrier(self, h, delta=50):
        if h < delta:
            return 0.5*(((h-2*delta)/delta)**2-1) - math.log(delta)
        else:
            return -math.log(h)

    def get_action_variant(self, state, t_curr):

        def cost(*A):
            # print(A)
            A = np.array(A)
            return -self.traj_reward(state, A, t_curr)
        scalars = [ng.p.Scalar(init=0, lower=-1, upper=1) for _ in range(self.horizon_step)]
        instrum = ng.p.Instrumentation(*scalars)
        # optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=5, num_workers=1)
        optimizer = ng.optimizers.BO(parametrization=instrum, budget=10, num_workers=1)
        A_star = optimizer.minimize(cost).value  # best value
        print("A: ", A_star)


        return A_star[0][0]



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
    Y = np.zeros(6)
    sigma = np.zeros(2)

    for t in tqdm(range(len(time))):

        action = agent.get_action(s, t, Y, sigma)
        # print("ACTION: ", action)
        s_, s_d, s_dd = plant.step(action)

        lamda = 1
        # pdb.set_trace()
        sigma = np.array([[(s_d[0] - 0) + lamda*(s_[0] - ref_x), (s_d[1] - 0) + lamda*(s_[1] - ref_x)]]) # 2D
        Y = np.array([s_dd[0], s_dd[1], s_d[0], s_d[1], s_[0], s_[1]])
        Y_T = np.array([Y]).T
        gamma = 0.002
        # pdb.set_trace()
        pi_d = gamma*np.matmul(Y_T, sigma)
        # pdb.set_trace()
        agent.update_model(pi_d, Y)

        s_list.append(s_)
        a_list.append(action)


    plant = Plant()
    s = plant.reset()
    agent = MPC()


    # s_list_nadapt = []
    # a_list_nadapt = []
    # Y = np.zeros(6)
    # sigma = np.zeros(2)

    # for t in tqdm(range(len(time))):

    #     action = agent.get_action(s, t, Y, sigma)
    #     # print("ACTION: ", action)
    #     s_, s_d, s_dd = plant.step(action)

    #     lamda = 1
    #     # pdb.set_trace()
    #     sigma = np.array([[(s_d[0] - 0) + lamda*(s_[0] - ref_x), (s_d[1] - 0) + lamda*(s_[1] - ref_x)]]) # 2D
    #     Y = np.array([s_dd[0], s_dd[1], s_d[0], s_d[1], s_[0], s_[1]])
    #     Y_T = np.array([Y]).T
    #     gamma = 0.002
    #     pi_d = gamma*np.matmul(Y_T, sigma)
    #     # agent.update_model(pi_d, Y)

    #     s_list_nadapt.append(s_)
    #     a_list_nadapt.append(action)

    # pdb.set_trace()

    s_list = np.array(s_list)
    # s_list_nadapt = np.array(s_list_nadapt)

    fig, ax = plt.subplots()

    cmap = plt.get_cmap("tab10")


    ax.plot(time, s_list[:, 0], color=cmap(0), label='m1 Displacement')
    ax.plot(time, s_list[:, 1], color=cmap(1), label='m2 Displacement')
    ax.plot(time, a_list, color=cmap(2), label="Action")
    # ax.plot(time, s_list_nadapt[:, 0], color=cmap(0), label='m1 Displacement (without adaptation)', linestyle='--')
    # ax.plot(time, s_list_nadapt[:, 1], color=cmap(1), label='m2 Displacement (without adaptation)', linestyle='--')
    ax.plot(time, agent.ref[:len(time), 0], linestyle='--', color=cmap(3), label='Reference')
    # ax.plot(time, agent.ref[:len(time), 1], linestyle='--', color=cmap(1), label='m2 Reference')

    ax.set(xlabel='time (s)', ylabel='x1, x2 (m)', title='State of the System')
    ax.legend(loc='lower right')
    ax.grid()
    plt.savefig("result.pdf")

    plt.show()