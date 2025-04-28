import copy
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import qmc
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################################################   
####################Physics-Informed Neural Network #############
NUM_EPOCHS = 100000
LEARNING_RATE = 1e-4
NUM_COLLOCATION = 10000
PATIENCE = 1000
THRESHOLD = 1e-3
EARLY_STOPPING_EPOCH = 50000
NUM_SAMPLES = 3000

T_START = 0.0
T_END = 5.0
dt = 0.1

X_MIN = 3.0
X_MAX = 30.0
S_MIN = 0.01
S_MAX = 2.0
V_MIN = 1.0
V_MAX = 3.0
F_MIN = 0.0
F_MAX = 0.1

# --- Model Parameters ---
# MU_MAX = 0.86980    # 1/h
# K_S = 0.000123762    # g/l
# Y_XS = 0.435749      # g/g
S_F = 286           # g/l

# PINN estimated kinetic parameters
MU_MAX = 0.8308
K_S    = 0.1004
Y_XS   = 0.3684

# Initial Conditions
X_0, S_0, V_0 = 5, 0.013, 1.7  # Biomass, Substrate, Volume

# ODE solver parameters
ODE_SOLVER = 'LSODA'
def numpy_to_tensor(array, requires_grad=False):
    return torch.tensor(array, requires_grad=requires_grad, dtype=torch.float32).to(DEVICE).reshape(-1, 1)

def grad(outputs, inputs):
    return torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]

class PINN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input = nn.Linear(input_dim, 64)
        self.hidden1 = nn.Linear(64, 1024)
        self.hidden  = nn.Linear(1024, 1024)
        self.hidden2 = nn.Linear(1024, 64)
        self.output = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = torch.tanh(self.input(x))
        x = torch.tanh(self.hidden1(x))
        x = torch.tanh(self.hidden(x))
        x = torch.tanh(self.hidden2(x))
        x = self.output(x)
        return x
def generate_dataset(num_samples: int = NUM_SAMPLES, sampling_method: str = 'lhs') -> tuple:
    """Generate dataset of random multiple initial conditions and control actions"""
    df = pd.DataFrame(columns=['t', 'X', 'S', 'V', 'F'])
    if sampling_method == 'uniform':
        df['X'] = np.random.uniform(X_MIN, X_MAX, num_samples)
        df['S'] = np.random.uniform(S_MIN, S_MAX, num_samples)
        df['V'] = np.random.uniform(V_MIN, V_MAX, num_samples)
        df['F'] = np.random.uniform(F_MIN, F_MAX, num_samples)
        df['t'] = 0.0 # initial time (always 0)
    elif sampling_method == 'lhs':
        sampler = qmc.LatinHypercube(d=4)
        lhs_samples = sampler.random(n=num_samples)
        scaled_samples = qmc.scale(lhs_samples, [X_MIN, S_MIN, V_MIN, F_MIN], [X_MAX, S_MAX, V_MAX, F_MAX])
        df = pd.DataFrame(scaled_samples, columns=['X', 'S', 'V', 'F'])
        df['t'] = 0.0
        
    t_train = numpy_to_tensor(df['t'].values, requires_grad=True)
    X_train = numpy_to_tensor(df['X'].values)
    S_train = numpy_to_tensor(df['S'].values)
    V_train = numpy_to_tensor(df['V'].values)
    F_train = numpy_to_tensor(df['F'].values)
    
    in_train = torch.cat([t_train, X_train, S_train, V_train, F_train], dim=1)
    out_train = torch.cat([X_train, S_train, V_train], dim=1)
    
    return in_train, out_train

def loss_fn(net: nn.Module, sampling_method: str = 'lhs') -> torch.Tensor:
    if sampling_method == 'uniform':
        t_col = numpy_to_tensor(np.random.uniform(T_START, dt, NUM_COLLOCATION), requires_grad=True)
        X0_col = numpy_to_tensor(np.random.uniform(X_MIN, X_MAX, NUM_COLLOCATION))
        S0_col = numpy_to_tensor(np.random.uniform(S_MIN, S_MAX, NUM_COLLOCATION))
        V0_col = numpy_to_tensor(np.random.uniform(V_MIN, V_MAX, NUM_COLLOCATION))
        F_col  = numpy_to_tensor(np.random.uniform(F_MIN, F_MAX, NUM_COLLOCATION))
    elif sampling_method == 'lhs':
        sampler = qmc.LatinHypercube(d=5)
        lhs_samples = sampler.random(n=NUM_COLLOCATION)
        scaled_samples = qmc.scale(lhs_samples, [T_START, X_MIN, S_MIN, V_MIN, F_MIN], [dt, X_MAX, S_MAX, V_MAX, F_MAX])
        t_col = numpy_to_tensor(scaled_samples[:, 0], requires_grad=True)
        X0_col = numpy_to_tensor(scaled_samples[:, 1])
        S0_col = numpy_to_tensor(scaled_samples[:, 2])
        V0_col = numpy_to_tensor(scaled_samples[:, 3])
        F_col  = numpy_to_tensor(scaled_samples[:, 4])
    
    u_col = torch.cat([t_col, X0_col, S0_col, V0_col, F_col], dim=1)
    preds = net.forward(u_col)

    X_pred = preds[:, 0].view(-1, 1)
    S_pred = preds[:, 1].view(-1, 1)
    # V_pred = preds[:, 2].view(-1, 1)
    V_pred = V0_col.view(-1, 1) + F_col.view(-1, 1) * t_col.view(-1, 1) # not used

    dXdt_pred = grad(X_pred, t_col)
    # dSdt_pred = grad(S_pred, t_col) 
    # dVdt_pred = grad(V_pred, t_col) # not used

    mu = MU_MAX * S_pred / (K_S + S_pred)

    # residuals
    rhs_X = mu * X_pred - (F_col / V_pred) * X_pred
    rhs_S = - (mu * X_pred) / Y_XS + (F_col / V_pred) * (S_F - S_pred)
    # rhs_V = F_col

    error_dXdt = dXdt_pred - rhs_X
    error_dSdt = 0.0 - rhs_S
    # error_dVdt = dVdt_pred - rhs_V

    # average residual squared loss
    w_X, w_S, w_V = 0.8, 0.2, 0.0
    loss_ode = torch.mean(error_dXdt**2) * w_X + \
               torch.mean(error_dSdt**2) * w_S 

    return loss_ode

in_train, out_train = generate_dataset()

print(f'Input shape: {in_train.shape}')
print(f'Output shape: {out_train.shape}')
model_name = "pinc_model_br.pth"
model_exists = os.path.exists(f'./models/{model_name}')

if model_exists:
    # Load the model
    net = PINN(input_dim=in_train.shape[1], output_dim=out_train.shape[1]).to(DEVICE)
    net.load_state_dict(torch.load(f'./models/{model_name}', weights_only=True))
    net.eval()
else:
    # Main
    net = PINN(input_dim=in_train.shape[1], output_dim=out_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.75)

    # Loss weights
    w_data, w_ode, w_ic = 1, 1, 0.0

    # Initialize early stopping variables
    best_loss = float("inf")
    best_model_weights = None
    patience = PATIENCE
    threshold = THRESHOLD
    pretrain_epochs = 0
    
    for epoch in tqdm(range(NUM_EPOCHS)):
        optimizer.zero_grad()
        preds = net.forward(in_train)
        X_pred = preds[:, 0].view(-1, 1)
        S_pred = preds[:, 1].view(-1, 1)
        V_pred = preds[:, 2].view(-1, 1)
        
        w_X, w_S, w_V = 0.7, 0.2, 0.1
        loss_data = torch.mean((X_pred - out_train[:, 0].view(-1, 1))**2) * w_X + \
                    torch.mean((S_pred - out_train[:, 1].view(-1, 1))**2) * w_S + \
                    torch.mean((V_pred - out_train[:, 2].view(-1, 1))**2) * w_V
        
        if epoch < pretrain_epochs:
            loss = loss_data * w_data
        else:
            loss_ode = loss_fn(net)
            loss = w_data * loss_data + w_ode * loss_ode
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 1000 == 0:
            if epoch < pretrain_epochs:
                print(f"Epoch {epoch}: Loss = {loss_data.item():.4f}")
            else:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Data Loss = {loss_data.item():.4f}, ODE Loss = {loss_ode.item():.4f}")
        
        # Early stopping
        if epoch >= EARLY_STOPPING_EPOCH:
            if loss.item() < best_loss - threshold:
                best_loss = loss.item()
                best_model_weights = copy.deepcopy(net.state_dict())
                patience = PATIENCE
            else:
                patience -= 1
                if patience <= 0:
                    print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.4f} at epoch {epoch - PATIENCE}.")
                    break

    # Load best model weights
    if best_model_weights is not None:
        net.load_state_dict(best_model_weights)
        net.eval()
        print("Loaded best model weights.")
    else:
        print("No model weights to load.")
        net.eval()

    # Save the model
    torch.save(net.state_dict(), f'./models/{model_name}')

