import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    time = np.arange(0, 40, 0.01)

    s_list = np.load("data/s_list.npy")
    s_list_nadapt = np.load("data/s_list_nadapt.npy")
    ref = np.load("data/ref.npy")
    param_list = np.load("data/param_list.npy", allow_pickle=True)
    param_star = np.load("data/param_star.npy")
    a_list = np.load("data/a_list.npy")
    a_n_list = np.load("data/a_n_list.npy")
    a_u_list = np.load("data/a_u_list.npy")
    costs = np.load("data/costs.npy")
    sigmas = np.load("data/sigmas.npy")

    sns.set_theme()

    fig, ax = plt.subplots(figsize=(10,5))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, left=0.1, right=1-0.06, top=1-0.08, hspace=0.3)
    cmap = plt.get_cmap("tab10")
    ax.plot(time, s_list[:, 0], color=cmap(0), label='$m_1$ Displacement')
    ax.plot(time, s_list[:, 1], color=cmap(1), label='$m_2$ Displacement')
    # ax.plot(time, a_list, color=cmap(2), label="Input $F$")
    ax.plot(time, s_list_nadapt[:, 0], color=cmap(0), label='$m_1$ Displacement (without adaptation)', linestyle='--')
    ax.plot(time, s_list_nadapt[:, 1], color=cmap(1), label='$m_2$ Displacement (without adaptation)', linestyle='--')
    ax.plot(time, ref, linestyle='--', color=cmap(3), label='Reference')
    # ax.plot(time, agent.ref[:len(time), 1], linestyle='--', color=cmap(1), label='m2 Reference')

    ax.set(xlabel='time (s)', ylabel='x1, x2 (m)', title='States of the System')
    ax.legend(loc='center right')
    # ax.grid()
    plt.savefig("pro.pdf")



    fig, ax = plt.subplots(figsize=(6.5,4.5))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, left=0.12, top=1-0.08, hspace=0.3)
    for i, lab, true_value in zip(range(6), ['m_1', 'm_2', 'k_1', 'k_2', 'b_1', 'b_2'], param_star):
        ax.plot(time, param_list[:, i], color=cmap(i), label=f"${lab}$")
        ax.plot(time, [true_value]*len(time), color=cmap(i), label=f"${lab}^*$", linestyle='--')
    ax.set(xlabel='time (s)', ylabel='Value', title='Parameters of the System')
    ax.legend(ncol=2, loc='upper right')
    # ax.grid()
    plt.savefig("param.pdf")

    fig, ax = plt.subplots(figsize=(6.5,4.5))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, left=0.12, top=1-0.08, hspace=0.3)
    ax.plot(time, a_list, color=cmap(0), label="Adaptive Input $F$")
    ax.plot(time, a_n_list, color=cmap(1), label="$\\tau_n$")
    ax.plot(time, a_u_list, color=cmap(2), label="$Y\\hat{\\pi}_u$")
    ax.set(xlabel='time (s)', ylabel='Force', title='Adaptive Force')
    ax.legend(loc='upper right')
    # ax.grid()
    plt.savefig("force.pdf")

    # fig, ax = plt.subplots(figsize=(6.5,4.5))
    # fig.tight_layout()
    # fig.subplots_adjust(bottom=0.12, left=0.14, top=1-0.08, hspace=0.3)
    # ax.plot(time, costs)
    # ax.set(xlabel='time (s)', ylabel='Root Mean Square Error (RMSE)', title='Error of States')
    # # ax.grid()
    fig, ax = plt.subplots(figsize=(6.5,4.5))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, left=0.12, top=1-0.08, hspace=0.3)
    ax.plot(time, sigmas[:,0], label="Composite Error $\\sigma_0$")
    ax.plot(time, sigmas[:,1], label="Composite Error $\\sigma_1$")
    ax.set(xlabel='time (s)', ylabel='Error', title='Composite Error')
    ax.legend(loc='upper right')

    plt.savefig("sigma.pdf")
    plt.show()
