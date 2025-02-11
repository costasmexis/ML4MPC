from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.optimize import LinearConstraint, minimize, differential_evolution
from scipy.integrate import solve_ivp
from sklearn.base import BaseEstimator
from pyswarm import pso
from tqdm import tqdm

from .machinelearning import model_predict

# --- Parameters ---
dt = 1      # Time step
Cb1 = 24.9  # Concentration of stream 1
Cb2 = 0.1   # Concentration of stream 2
k1 = 1.0    # Reaction rate constant
k2 = 1.0    # Reaction parameter
N = 5      # Prediction horizon
L = 60      # Simulation steps
Q = 1.0     # Weight for tracking
R = 0.1     # Weight for control effort
h0 = 10.0   # Initial height
w1_min, w1_max = 0, 4.0  # Control input bounds (w1)
w2_min, w2_max = 0, 4.0  # Control input bounds (w2)
Cb_min, Cb_max = 20.0, 25.0  # Concentration bounds
h_min, h_max = 8.0, 12.0
delta_w1_max = 0.5  # Max rate of change for control input
Cb0 = 22.0  # Initial concentration
w2 = 0.1    # Disturbance input

# --- Real System Dynamics --- #
def system_of_odes(t, y, w1, w2):
    h, Cb = y
    dh_dt = w1 + w2 - 0.2 * np.sqrt(h)
    dCb_dt = ((Cb1 - Cb) * w1 / h + (Cb2 - Cb) * w2 / h - k1 * Cb / (1 + k2 * Cb)**2)
    return [dh_dt, dCb_dt]

# --- Generate training dataset  ---
def generate_training_data(samples: int=1000, return_df: bool=True):
    X, Y = [], []
    for _ in tqdm(range(samples)):
        w1 = np.random.uniform(w1_min, w1_max)
        # w2 = np.random.uniform(w2_min, w2_max)
        w2 = 0.1
        Cb = np.random.uniform(Cb_min, Cb_max)
        h = np.random.uniform(h_min, h_max)
        dCb_dt = ((Cb1 - Cb) * w1 / h + (Cb2 - Cb) * w2 / h - k1 * Cb / (1 + k2 * Cb)**2)
        Cb_next = Cb + dCb_dt * dt
        X.append([w1, w2, Cb])
        Y.append(Cb_next)
    if return_df:
        df = pd.DataFrame(X, columns=['w1', 'w2', 'Cb'])
        df['Cb_next'] = Y
        return df
    else:
        return np.array(X), np.array(Y)

# --- MPC --- #
# --- Cost Function --- #
def mpc_cost(w1_seq, Cb_ref, Cb0, w2, model):
    cost = 0
    Cb = Cb0 # Start from the actual system value
    for idx in range(N):
        w1 = w1_seq[idx]
        Cb = model_predict(Cb, w1, w2, model)
        cost += Q * (Cb_ref[idx] - Cb) ** 2
        if idx > 0:  # Penalize difference between consecutive control actions
            cost += R * (w1 - w1_seq[idx - 1]) ** 2
    return cost

# --- MPC Solver ---
def solve_mpc(Cb_ref, Cb, w1_ini, w2, model):

    delta_w1_matrix = np.eye(N) - np.eye(N, k=1)
    delta_w1_matrix = delta_w1_matrix[:-1, :]  # Remove the last row 
    rate_constraint = LinearConstraint(delta_w1_matrix, -delta_w1_max, delta_w1_max)
    bounds = [(w1_min, w1_max) for _ in range(N)]
    
    result = minimize(
        mpc_cost, 
        w1_ini,
        args=(Cb_ref, Cb, w2, model),
        bounds=bounds,
        constraints=[rate_constraint],
        method='COBYLA' # COBYLA   
    )

    if not result.success:
        print("Optimization failed, using default control inputs")
        return np.ones(N) * w1_min

    return result.x

def solve_mpc_ga(Cb_ref, Cb, w2, model):
    def mpc_cost_ga(w1_flat):
        return mpc_cost(w1_flat, Cb_ref, Cb, w2, model)
    bounds = [(w1_min, w1_max) for _ in range(N)]
    result = differential_evolution(mpc_cost_ga,bounds)
    if not result.success:
        print("Optimization failed, using default control inputs")
        return np.ones(N) * w1_min
    return result.x

def solve_mpc_pso(Cb_ref, Cb, w2, model, max_iter=100, swarm_size=50):
    def mpc_cost_pso(w1_flat):
        return mpc_cost(w1_flat, Cb_ref, Cb, w2, model)
    lb = [w1_min] * N
    ub = [w1_max] * N
    best_w1, _ = pso(
        mpc_cost_pso,
        lb,
        ub,
        swarmsize=swarm_size,
        maxiter=max_iter,
        debug=False
    )
    return best_w1

# --- Simulation Setup ---
def simulation(model: Union[BaseEstimator, nn.Module], Cb_ref: list, method: str = 'opt'):
    ''' Simulate the CSTR system using MPC 
    - method: 'opt' for optimization, 'GA' for genetic algorithm, 'PSO' for particle swarm optimization
    '''
    Cb = np.zeros(L + 1)
    Cb[0] = Cb0
    w1 = np.zeros(L)
    h = np.zeros(L + 1)
    h[0] = h0
    w1_ini = np.ones(N) * 0.25

    for idx in tqdm(range(L)):
        # Adjust the reference trajectory slice for the prediction horizon
        Cb_ref_slice = Cb_ref[idx:idx+N]
        if len(Cb_ref_slice) < N:
            Cb_ref_slice = np.append(Cb_ref_slice, [Cb_ref_slice[-1]] * (N - len(Cb_ref_slice)))

        # Solve MPC optimization problem
        if method == 'GA':
            w1_mpc = solve_mpc_ga(Cb_ref_slice, Cb[idx], w2, model)
        elif method == 'PSO':
            w1_mpc = solve_mpc_pso(Cb_ref_slice, Cb[idx], w2, model)
        else:
            w1_mpc = solve_mpc(Cb_ref_slice, Cb[idx], w1_ini, w2, model)

        # Apply the first control input
        w1[idx] = w1_mpc[0]
        w1_ini = np.append(w1_mpc[1:], w1_mpc[-1])  # Shift control sequence
        
        # Update system state using REAL SYSTEM (ODE solver)
        sol = solve_ivp(
            system_of_odes,
            t_span=[idx * dt, (idx + 1) * dt],
            y0=[h[idx], Cb[idx]],
            args=(w1[idx], w2),
            t_eval= [idx * dt, (idx + 1) * dt],
            method="Radau"
        )
        h[idx + 1], Cb[idx + 1] = sol.y[:, -1]
        
    return Cb, w1

# --- Plot Results ---
def plot_results(Cb, Cb_ref, w1):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(range(L + 1), Cb, label="True Concentration (Cb)", color="orange")
    plt.plot(range(L), Cb_ref, label="Set Point (Cb_ref)", linestyle="--", color="red")
    plt.xlabel("Time")
    plt.ylabel("Concentration (Cb)")
    plt.title("Concentration vs. Time")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.step(range(L), w1, where="post", label="Control Input (w1)", color="blue")
    plt.xlabel("Time")
    plt.ylabel("Inflow Rate (w1)")
    plt.title("Control Input vs. Time")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
    


