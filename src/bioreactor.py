import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.integrate import odeint
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000
MU_MAX = 0.20 # 1/h
K_S = 1       # g/l
Y_XS = 0.5    # g/g
Y_PX = 0.2    # g/g
S_F = 10      # g/l

T_START = 0
T_END = 100
TIME_RANGE = int(T_END - T_START)

# Initial conditions
X_0 = 0.05
S_0 = 10.0
V_0 = 1.0
F_0 = 1
IC = [0.05, 0.0, 10.0, 1.0]  # [X0, P0, S0, V0]

dt = 1                        # Time step
L = int(TIME_RANGE / dt)      # Simulation steps
N_p = 5                       # Prediction horizon
Q = 1.0                       # Weight for tracking
R = 0.1                       # Weight for control effort

# Bounds for feeding rate
F_MIN = 0.0                  # l/h
F_MAX = 2.0                  # l/h
BOUNDS = [(F_MIN, F_MAX) for _ in range(N_p)] 

# Dynamic model
def system_of_odes(t, y, F):
    X, S, V = y
    dX_dt = (MU_MAX * S / (K_S + S)) * X - (F / V) * X
    dS_dt = -(1 / Y_XS) * (MU_MAX * S / (K_S + S)) * X + (F / V) * (S_F - S)
    dV_dt = F
    return np.array([dX_dt, dS_dt, dV_dt])

# Discretized model
def discrete_model(X, S, V, F, h):
    k1 = system_of_odes(t, [X, S, V], F)
    k2 = system_of_odes(t + h / 2, [X + k1[0] * h / 2, S + k1[1] * h / 2, V + k1[2] * h / 2], F)
    k3 = system_of_odes(t + h / 2, [X + k2[0] * h / 2, S + k2[1] * h / 2, V + k2[2] * h / 2], F)
    k4 = system_of_odes(t + h, [X + k3[0] * h, S + k3[1] * h, V + k3[2] * h], F)

    X_next = X + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    S_next = S + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    V_next = V + (h / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return X_next, S_next, V_next