import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Parameters ---
dt = 1  # Time step
Cb1 = 24.9  # Concentration of stream 1
Cb2 = 0.1   # Concentration of stream 2
k1 = 1.0    # Reaction rate constant
k2 = 1.0    # Reaction parameter
N = 7    # Prediction horizon
L = 60      # Simulation steps
Q = 1.0     # Weight for tracking
R = 0.1     # Weight for control effort
w1_min, w1_max = 0, 4.0  # Control input bounds
Cb_min, Cb_max = 20.0, 25.0  # Concentration bounds
h_min, h_max = 8.0, 12.0
delta_w1_max = 0.5  # Max rate of change for control input
Cb0 = 22.0  # Initial concentration
w2 = 0.1    # Disturbance input

# --- Generate training dataset  ---
def generate_training_data(samples: int=1000, return_df: bool=True):
    X, Y = [], []
    for _ in tqdm(range(samples)):
        w1 = np.random.uniform(w1_min, w1_max)
        Cb = np.random.uniform(Cb_min, Cb_max)
        h = np.random.uniform(h_min, h_max)
        dCb_dt = ((Cb1 - Cb) * w1 / h + (Cb2 - Cb) * w2 / h -
                  k1 * Cb / (1 + k2 * Cb)**2)
        Cb_next = Cb + dCb_dt * dt
        X.append([w1, Cb])
        Y.append(Cb_next)
    if return_df:
        return pd.DataFrame(X, columns=['w1', 'Cb']).assign(Cb_next=Y)
    return np.array(X), np.array(Y)
