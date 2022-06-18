import os
import pdb
import torch
from botorch.test_functions import Hartmann
from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
import numpy as np
from matplotlib import pyplot as plt
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from tqdm import tqdm
import time
import warnings


warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
# neg_hartmann6 = Hartmann(negate=True)
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 512
NOISE_SE = 0.1


N_TRIALS = 3 
N_BATCH = 20
MC_SAMPLES = 256

def func(x):
    y = np.mean(x.numpy())
    return torch.as_tensor([y])


# def outcome_constraint(X):
#     """L1 constraint; feasible if less than or equal to zero."""
#     return X.sum(dim=-1) - 3

# def weighted_obj(X):
#     """Feasibility weighted objective; zero if not feasible."""
#     # pdb.set_trace()
#     return neg_hartmann6(X) * (outcome_constraint(X) <= 0).type_as(X)


def generate_initial_data(n, func, g, dim):
    # generate training data
    train_x = torch.rand(n, dim, device=device, dtype=dtype)
    exact_obj = func(train_x).unsqueeze(-1)  # add output dimension
    exact_con = g(train_x).unsqueeze(-1)  # add output dimension
    train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    train_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
    # best_observed_value = weighted_obj(train_x).max().item()
    return train_x, train_obj, train_con
    
    
def initialize_model(train_x, train_obj, train_con, state_dict=None):
    # define models for objective and constraint
    train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)
    # pdb.set_trace()
    model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
    model_con = FixedNoiseGP(train_x, train_con, train_yvar.expand_as(train_con)).to(train_x)
    # combine into a multi-output GP model
    model = ModelListGP(model_obj, model_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def obj_callable(Z):
    return Z[..., 0]

def constraint_callable(Z):
    return Z[..., 1]

def optimize_acqf_and_get_observation(acq_func, f, g, dim):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    bounds = torch.tensor([[0.0] * dim, [1.0] * dim], device=device, dtype=dtype)
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values 
    new_x = candidates.detach()
    exact_obj = f(new_x).unsqueeze(-1)  # add output dimension
    exact_con = g(new_x).unsqueeze(-1)  # add output dimension
    new_obj = exact_obj# + NOISE_SE * torch.randn_like(exact_obj)
    new_con = exact_con# + NOISE_SE * torch.randn_like(exact_con)
    return new_x, new_obj, new_con


def optimise(func, constraint, dim=6):


    def f(x):
        y = func(x.numpy().flatten())
        return torch.as_tensor([y])

    def g(x):
        y = constraint(x.numpy().flatten())
        return torch.as_tensor([y])
        



    # define a feasibility-weighted objective for optimization
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable],
    )

    verbose = False

    # best_observed = []

    # call helper functions to generate initial training data and initialize model
    train_x, train_obj, train_con = generate_initial_data(1, f, g, dim)
    mll, model = initialize_model(train_x, train_obj, train_con)
    # best_observed.append(best_observed_value)

    best_y_uncon, best_x_uncon = train_obj.item(), train_x.numpy().flatten()

    if train_con.item() <= 0:
        best_y = train_obj.item()
        best_x = train_x.numpy().flatten()
    else:
        best_y = float('-inf')
        best_x = None

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in tqdm(range(1, N_BATCH + 1)):    
        
        # t0 = time.time()
        
        # fit the models
        fit_gpytorch_model(mll)

        
        # define the qEI and qNEI acquisition modules using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
        
        # for best_f, we use the best observed noisy values as an approximation
        qEI = qExpectedImprovement(
            model=model, 
            best_f=(train_obj * (train_con <= 0).to(train_obj)).max(),
            sampler=qmc_sampler, 
            objective=constrained_obj,
        )
        
        # optimize and get new observation
        new_x, new_obj, new_con = optimize_acqf_and_get_observation(qEI, f, g, dim)

        if new_obj.item() >= best_y and new_con.item() <= 0:
            best_y = new_obj.item()
            best_x = new_x.numpy().flatten()

        if new_obj.item() >= best_y_uncon:
            best_y_uncon = new_obj.item()
            best_x_uncon = new_x.numpy().flatten()

                
        # update training points
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        train_con = torch.cat([train_con, new_con])

        # update progress
        # best_value = weighted_obj(train_x).max().item()
        # pdb.set_trace()
        # best_observed.append(best_value)



        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll, model = initialize_model(
            train_x, 
            train_obj, 
            train_con, 
            model.state_dict(),
        )

        # pdb.set_trace()

        
        t1 = time.time()
        
        # if verbose:
        #     print(
        #         f"\nBatch {iteration:>2}: best_value = {best_value:>4.2f}",
        #         f"time = {t1-t0:>4.2f}.", end=""
        #     )
        # else:
        #     print(".", end="")

        

    if best_x is None:
        best_y = best_y_uncon
        best_x = best_x_uncon
        print("FAIL TO MEET THE CONSTRAINT .....")
    
    if np.shape(best_x) != (1,):
        pdb.set_trace()

    return best_y, best_x



# # GLOBAL_MAXIMUM = neg_hartmann6.optimal_value
# GLOBAL_MAXIMUM = 0


# iters = np.arange(N_BATCH + 1) * BATCH_SIZE

# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# ax.errorbar(iters, best_observed, yerr=0.1, label="qEI", linewidth=1.5)
# plt.plot([0, N_BATCH * BATCH_SIZE], [GLOBAL_MAXIMUM] * 2, 'k', label="true best objective", linewidth=2)
# ax.set_ylim(bottom=0.5)
# ax.set(xlabel='number of observations (beyond initial points)', ylabel='best objective value')
# ax.legend(loc="lower right")

# plt.show()

# def func(x):
#     y = np.mean(x)
#     return y


# def const(x):

#     return np.std(x)


# best_y, best_x = optimise(func, const, 1)
# # print(best_y)