import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tqdm import tqdm

from pyswarm import pso
from scipy.optimize import differential_evolution

# ----- Constants -----
# Num of samples
NUM_SAMPLES = 1000

# Kinetic parameters
MU_MAX = 0.870       # 1/h
K_S    = 0.215       # g/l
Y_XS   = 0.496       # g/g
Y_PX   = 0.2         # g/g
S_F    = 1.43 * 200  # g/l

# Initial conditions
X_0 = 5.85
S_0 = 0.013
V_0 = 1.56

# Time parameters
T_START = 0
T_END = 5
TIME_RANGE = int(T_END - T_START) # Absolute time 

# MPC parameters
dt = 0.1                      # Time step
L = int(TIME_RANGE / dt)      # Simulation steps
N_p = 7                       # Prediction horizon
Q = 1.0                       # Weight for tracking
R = 0.8                       # Weight for control effort
OPTIMIZATION_METHOD = 'SLSQP' # Optimization method. Other options: 'SLSQP, 'L-BFGS-B', 'trust-constr', 'COBYLA', 'Powell', 'Nelder-Mead'

# Bounds for feeding rate
F_MIN = 0.0                  # l/h
F_MAX = 0.1                  # l/h
F_0 = (F_MAX + F_MIN) / 2    # Initial feed rate
BOUNDS = [(F_MIN, F_MAX) for _ in range(N_p)] 

############## Simulate system using ODEs / kinetic parameters / IC #############
def plant_model(t, y, F_func: callable):
    X, S, V = y
    F = F_func(t)
    dX_dt = (MU_MAX * S / (K_S + S)) * X - (F / V) * X
    dS_dt = -(1 / Y_XS) * (MU_MAX * S / (K_S + S)) * X + (F / V) * (S_F - S)
    dV_dt = F
    return [dX_dt, dS_dt, dV_dt]

def simulate(F: callable, plot: bool=True) -> np.ndarray:
    """ Simulate bioreactor system using ODEs 
    - F: feed rate (assumed constant)
    """
    t_points = np.arange(T_START, T_END, dt)
    F_func = interp1d(t_points, [F(t) for t in t_points], kind="linear", fill_value="extrapolate")
    
    sol = solve_ivp(plant_model, t_span=(T_START, T_END), y0=[X_0, S_0, V_0], args=(F_func,), t_eval=np.arange(T_START, T_END, dt))
    if plot:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(sol.t, sol.y[0], label='Biomass')
        plt.plot(sol.t, sol.y[1], label='Substrate')
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(sol.t, sol.y[0], label='Biomass')
        ax1.plot(sol.t, sol.y[1], label='Substrate')
        ax1.set_title('Bioreactor Simulation')
        ax1.legend()
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.step(sol.t, [F(t) for t in sol.t], label='Feed Rate', color='gray', linestyle='--')
        ax2.set_ylabel('Feed Rate')
        ax2.legend(loc='upper right')
        plt.title('Bioreactor Simulation')
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(sol.t, sol.y[2], label='Volume')
        plt.legend()
        plt.grid()

        plt.show()
    return sol

############# Discretized model #############
# ----- Bioreactor model -----
def dynamic_system(t, y, F):
    X, S, V = y
    dX_dt = (MU_MAX * S / (K_S + S)) * X - (F / V) * X
    dS_dt = -(1 / Y_XS) * (MU_MAX * S / (K_S + S)) * X + (F / V) * (S_F - S)
    dV_dt = F
    return np.array([dX_dt, dS_dt, dV_dt])

def discretized_model(t, X, S, V, F, h=0.01):
    k1 = dynamic_system(t, [X, S, V], F)
    k2 = dynamic_system(t + h / 2, [X + k1[0] * h / 2, S + k1[1] * h / 2, V + k1[2] * h / 2], F)
    k3 = dynamic_system(t + h / 2, [X + k2[0] * h / 2, S + k2[1] * h / 2, V + k2[2] * h / 2], F)
    k4 = dynamic_system(t + h, [X + k3[0] * h, S + k3[1] * h, V + k3[2] * h], F)

    X_next = X + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    S_next = S + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    V_next = V + (h / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return X_next, S_next, V_next

############# Model Predictive Control #############
# ----- Set-point trajectory func -----
def set_point(t):
    # if t <= 2.5:
    #     return 10
    # elif 2.5 <= t < 5:
    #     return 15
    # else:
    #     return 20
    return X_0 * np.exp(0.3 * t)


# ----- Cost function -----
def cost_function(F_opt, X, S, V, t, model='discretized'):
    """ Cost function for MPC 
    - model: 'discretized' or a trained sklearn model
    """
    J = 0
    X_curr, S_curr, V_curr = X, S, V
    for k in range(N_p):
        X_sp = set_point(t + k * dt)
        if model == 'discretized':
            X_next, S_next, V_next = discretized_model(t, X_curr, S_curr, V_curr, F_opt[k])
        elif hasattr(model, 'predict'):
            # PyTorch Neural Network
            X_next, S_next, V_next = model.predict([[X_curr, S_curr, V_curr, F_opt[k]]])[0]
        elif hasattr(model, 'forward'):
            # Physics-informed neural network
            preds = model.forward(torch.tensor(np.array([dt, X_curr, S_curr, V_curr, F_opt[k]]), dtype=torch.float32).to(DEVICE))
            X_next = preds[0].detach().cpu().numpy()
            S_next = preds[1].detach().cpu().numpy()
            V_next = preds[2].detach().cpu().numpy()
        J += Q * (X_sp - X_next) ** 2
        if k > 0:
            J += R * (F_opt[k] - F_opt[k - 1]) ** 2
        X_curr, S_curr, V_curr = X_next, S_next, V_next
    return J

# ----- MPC -----
def mpc(model: str = 'discretized'):
    # Initialize system variables
    X = np.ones(L+1)
    S = np.ones(L+1)
    V = np.ones(L+1)
    F = np.ones(L)
    X[0], S[0], V[0] = X_0, S_0, V_0
    
    # MPC Loop
    for step in tqdm(range(L)):
        t=step*dt
        res = minimize(cost_function, F_0 * np.ones(N_p), args=(X[step], S[step], V[step], t, model), bounds=BOUNDS, method=OPTIMIZATION_METHOD)
        F[step] = res.x[0]
        sol = solve_ivp(dynamic_system, t_span=(t, t + dt), y0=[X[step], S[step], V[step]], args=(F[step],))
        X[step + 1], S[step + 1], V[step + 1] = sol.y[:, -1]

    return X, S, V, F

# ----- MPC with PSO -----
def mpc_pso(model: str = 'discretized'):
    # Initialize system variables
    X = np.ones(L+1)
    S = np.ones(L+1)
    V = np.ones(L+1)
    F = np.ones(L)
    X[0], S[0], V[0] = X_0, S_0, V_0
    
    # Define the cost function for GA
    def ga_cost_function(F_opt, X_current, S_current, V_current, t_current, model):
        return cost_function(F_opt, X_current, S_current, V_current, t_current, model)
    
    # MPC Loop
    for step in tqdm(range(L)):
        t = step * dt
        # Use GA to optimize the cost function
        F_opt, _ = pso(ga_cost_function, lb=[F_MIN]*N_p, ub=[F_MAX]*N_p, args=(X[step], S[step], V[step], t, model))
        F[step] = F_opt[0]
        sol = solve_ivp(dynamic_system, t_span=(t, t + dt), y0=[X[step], S[step], V[step]], args=(F[step],))
        X[step + 1], S[step + 1], V[step + 1] = sol.y[:, -1]

    return X, S, V, F

# ----- MPC with differential evolution -----
def mpc_diff_evol(model: str = 'discretized'):
    # Initialize system variables
    X = np.ones(L + 1)
    S = np.ones(L + 1)
    V = np.ones(L + 1)
    F = np.ones(L)
    X[0], S[0], V[0] = X_0, S_0, V_0

    # MPC Loop
    for step in tqdm(range(L)):
        t = step * dt

        # Define the bounds for F
        bounds = [(F_MIN, F_MAX) for _ in range(N_p)]  # BOUNDS should be a list of tuples, e.g., [(F_min, F_max)]

        # Use Differential Evolution to optimize the cost function
        res = differential_evolution(
            cost_function,  # Cost function to minimize
            bounds,  # Bounds for F
            args=(X[step], S[step], V[step], t, model),  # Additional arguments for cost_function
            strategy='best1bin',  # Mutation strategy
            maxiter=50,  # Maximum number of iterations
            popsize=10,  # Population size
            tol=1e-6,  # Tolerance for convergence
            mutation=(0.5, 1),  # Mutation factor range
            recombination=0.7,  # Crossover probability
            seed=42  # Random seed for reproducibility
        )

        # Extract the optimized F value
        F[step] = res.x[0]

        # Integrate the system dynamics
        sol = solve_ivp(dynamic_system, t_span=(t, t + dt), y0=[X[step], S[step], V[step]], args=(F[step],))
        X[step + 1], S[step + 1], V[step + 1] = sol.y[:, -1]

    return X, S, V, F

# ----- Plot MPC results -----
def plot_results(X, F):
    times = np.arange(0, TIME_RANGE+1, dt)
    SP = [set_point(t) for t in times]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, TIME_RANGE+dt, dt), X, label='Biomass Concentration')
    plt.plot(times, SP, 'r--', label='Setpoint')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.step(np.arange(0, TIME_RANGE, dt), F, label='Feed Rate')
    plt.legend()
    plt.grid()

    plt.show()

############# Evaluation #############
# -------- Evaluate MPC solving system of ODEs --------
def evaluation(F):
    t_points = np.arange(0, TIME_RANGE, dt)
    F_func = interp1d(t_points, F, kind="linear", fill_value="extrapolate")

    y0 = [X_0, S_0, V_0]
    times = np.arange(0, TIME_RANGE+1, dt)
    sol_t = [0]
    sol_X = [X_0]
    sol_S = [S_0]
    sol_V = [V_0]

    for i in range(len(times)-1):
        sol = solve_ivp(plant_model, t_span=[times[i], times[i+1]], y0=y0, args=(F_func,), method='RK45')
        y0 = sol.y[:, -1]  # Update initial condition
        sol_t.append(times[i+1])
        sol_X.append(y0[0])
        sol_S.append(y0[1])
        sol_V.append(y0[2])
        
    sol_t = np.array(sol_t)
    sol_X = np.array(sol_X)
    sol_S = np.array(sol_S)
    sol_V = np.array(sol_V)

    plt.figure(figsize=(12, 18))
    plt.subplot(3, 1, 1)
    plt.plot(times, [set_point(t) for t in times], "r--", label="Setpoint")
    plt.plot(sol_t, sol_X, label='Biomass Concentration')
    plt.plot(sol_t, sol_S, label='Substrate Concentration')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.step(sol_t, F_func(sol_t), label='Feed Rate')
    plt.legend()
    plt.grid()
    
    plt.subplot(3, 1, 3)
    plt.plot(sol_t, sol_V, label='Volume')
    plt.legend()
    plt.grid()

    plt.show()
    
    
    
############# Machine Learning #############
# -------- Generate training data --------
def generate_training_data():

    def system_odes(t, y, F):
        X, S, V = y
        dX_dt = (MU_MAX * S / (K_S + S)) * X - (F / V) * X
        dS_dt = -(1 / Y_XS) * (MU_MAX * S / (K_S + S)) * X + (F / V) * (S_F - S)
        dV_dt = F
        return [dX_dt, dS_dt, dV_dt]

    t_span = [0, TIME_RANGE]
    X, y = [], []
    for _ in range(NUM_SAMPLES):
        X0 = np.random.uniform(0.1, 5)
        S0 = np.random.uniform(5, 30)
        V0 = np.random.uniform(0.5, 2)
        F =  np.random.uniform(0.5, 2)
        
        y0 = [X0, S0, V0]
        
        sol = solve_ivp(system_odes, t_span, y0, args=(F,), t_eval=np.arange(t_span[0], t_span[1], dt))
        
        for i in range(len(sol.t) - 1):
            X_t = [sol.y[0, i], sol.y[1, i], sol.y[2, i], F]  # Features
            y_t = [sol.y[0, i + 1], sol.y[1, i + 1], sol.y[2, i + 1]]  # Target
            
            X.append(X_t)
            y.append(y_t)
            
    return np.array(X), np.array(y)

   
####################Physics-Informed Neural Network #############
NUM_EPOCHS = 50000
LEARNING_RATE = 1e-2
NUM_COLLOCATION = 10000
PATIENCE = 1000
THRESHOLD = 1e-3
EARLY_STOPPING_EPOCH = 1
NUM_SAMPLES = 1000

X_MIN = 0.01
X_MAX = 15.0
S_MIN = 0.10
S_MAX = 20.0
V_MIN = 0.5
V_MAX = 5.0
F_MIN = 0.01
F_MAX = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def numpy_to_tensor(array):
    return torch.tensor(array, requires_grad=True, dtype=torch.float32).to(DEVICE).reshape(-1, 1)

def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

class PINN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PINN, self).__init__()
        self.input = nn.Linear(input_dim, 64)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 64)
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.output(x)
        return x
    
def generate_dataset(num_samples: int = NUM_SAMPLES):
    """Generate dataset of random multiple initial conditions and control actions"""
    df = pd.DataFrame(columns=['t', 'X', 'S', 'V'])
    df['X'] = np.random.uniform(X_MIN, X_MAX, num_samples)
    df['S'] = np.random.uniform(S_MIN, S_MAX, num_samples)
    df['V'] = np.random.uniform(V_MIN, V_MAX, num_samples)
    df['F'] = np.random.uniform(F_MIN, F_MAX, num_samples)
    df['t'] = 0.0 # initial time
    
    t_train = numpy_to_tensor(df['t'].values)
    X_train = numpy_to_tensor(df['X'].values)
    S_train = numpy_to_tensor(df['S'].values)
    V_train = numpy_to_tensor(df['V'].values)
    F_train = numpy_to_tensor(df['F'].values)
    
    in_train = torch.cat([t_train, X_train, S_train, V_train, F_train], dim=1)
    out_train = torch.cat([X_train, S_train, V_train], dim=1)
    
    return in_train, out_train

def loss_fn(net: nn.Module) -> torch.Tensor:
    t_col = numpy_to_tensor(np.random.uniform(T_START, dt, NUM_COLLOCATION))
    X0_col = numpy_to_tensor(np.random.uniform(X_MIN, X_MAX, NUM_COLLOCATION))
    S0_col = numpy_to_tensor(np.random.uniform(S_MIN, S_MAX, NUM_COLLOCATION))
    V0_col = numpy_to_tensor(np.random.uniform(V_MIN, V_MAX, NUM_COLLOCATION))
    F_col = numpy_to_tensor(np.random.uniform(F_MIN, F_MAX, NUM_COLLOCATION))
    
    u_col = torch.cat([t_col, X0_col, S0_col, V0_col, F_col], dim=1)

    preds = net.forward(u_col)

    X_pred = preds[:, 0].view(-1, 1)
    S_pred = preds[:, 1].view(-1, 1)
    V_pred = preds[:, 2].view(-1, 1)

    dXdt_pred = grad(X_pred, t_col)[0]
    dSdt_pred = grad(S_pred, t_col)[0]
    dVdt_pred = grad(V_pred, t_col)[0]

    mu = MU_MAX * S_pred / (K_S + S_pred)

    error_dXdt = dXdt_pred - mu * X_pred + X_pred * F_col / V0_col
    error_dSdt = dSdt_pred + mu * X_pred / Y_XS - F_col / V0_col * (S_F - S_pred)
    error_dVdt = dVdt_pred - F_col
    
    error_ode = 1/3 * torch.mean(error_dXdt**2) + 1/3 * torch.mean(error_dSdt**2) + 1/3 * torch.mean(error_dVdt**2)

    return error_ode

def main(in_train: torch.Tensor, out_train: torch.Tensor, verbose: int = 100):
    
    net = PINN(input_dim=5, output_dim=3).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.7)

    # Loss weights
    w_data, w_ode, w_ic = 1.0, 1.0, 1.0

    # Initialize early stopping variables
    best_loss = float("inf")
    best_model_weights = None
    patience = PATIENCE
    threshold = THRESHOLD

    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()
        preds = net.forward(in_train)
        X_pred = preds[:, 0].view(-1, 1)
        S_pred = preds[:, 1].view(-1, 1)
        V_pred = preds[:, 2].view(-1, 1)
        loss_X = nn.MSELoss()(X_pred, out_train[:, 0].view(-1, 1))
        loss_S = nn.MSELoss()(S_pred, out_train[:, 1].view(-1, 1))
        loss_V = nn.MSELoss()(V_pred, out_train[:, 2].view(-1, 1))
        loss_data = 0.33 * (loss_X + loss_S + loss_V)

        loss_ode = loss_fn(net)

        loss = w_data * loss_data + w_ode * loss_ode
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % verbose == 0:
            print(f"Epoch {epoch}, Loss_data: {loss_data.item():.4f}, Loss_ode: {loss_ode.item():.4f}")
            # Print the current learning rate of the optimizer
            for param_group in optimizer.param_groups:
                print("Current learning rate: ", param_group["lr"])

        if epoch >= EARLY_STOPPING_EPOCH:
            if loss < best_loss - threshold:
                best_loss = loss
                best_model_weights = copy.deepcopy(net.state_dict())
                patience = 1000
            else:
                patience -= 1
                if patience == 0:
                    print(f"Early stopping at epoch {epoch}")
                    net.load_state_dict(best_model_weights)
                    break

    return net


