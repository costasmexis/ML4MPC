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
T_END = 60
IC = [0.05, 0.0, 10.0, 1.0]  # [X0, P0, S0, V0]
F_0 = 0.05                   # l/h

dt = 1
N = 5       # Prediction horizon
L = 60      # Simulation steps
Q = 1.0     # Weight for tracking
R = 0.1     # Weight for control effort

def feeding(t: float) -> float:
    ''' Feed rate as a function of time '''
    return F_0 * np.exp(0.05 * t)

def plot_simulation(df: pd.DataFrame) -> None:
    ''' Plot the simulation results '''
    plt.figure(figsize=(12, 3))
    plt.plot(df.t, df.X, linestyle='-', marker='o', markersize=3, label='Biomass')
    plt.plot(df.t, df.P, linestyle='-', marker='o', markersize=3, label='Product')
    plt.plot(df.t, df.S, linestyle='-', marker='o', markersize=3, label='Substrate')
    plt.plot(df.t, df.V, linestyle='-', marker='o', markersize=3, label='Volume (L)')
    plt.legend()
    plt.xlabel('Time (h)')
    plt.ylabel('Concentration (g/l)')
    plt.title('Bioreactor Simulation')
    plt.show()

def plot_feeding() -> None:
    ''' Plot the feeding rate over time '''
    t = np.linspace(T_START, T_END, NUM_SAMPLES)
    plt.figure(figsize=(12, 3))
    plt.step(t, feeding(t))
    plt.xlabel('Time (h)')
    plt.ylabel('Feed rate (l/h)')
    plt.title('Feed rate over time')
    plt.show()

def mu(S: float, mu_max: float = MU_MAX, K_s: float = K_S) -> float:
    return mu_max * S / (K_s + S)
    
def simulate(mu_max: float = MU_MAX, K_s: float = K_S, Y_xs: float = Y_XS, Y_px: float = Y_PX,\
            t_start: float = T_START, t_end: float = T_END, S_f: float = S_F, y0: list = IC) -> pd.DataFrame:
    """ Simulate the bioreactor model 
    - IC = [X0, P0, S0, V0] -> initial conditions
    """
    def bioreactor_model(y, t):
        X, P, S, V = y
        dX = mu(S, mu_max, K_s) * X - feeding(t) * X / V
        dS = -mu(S, mu_max, K_s) * X / Y_xs + feeding(t) * (S_f - S) / V
        dP = Y_px * mu(S, mu_max, K_s) * X - feeding(t) * P / V
        dV = feeding(t)
        return [dX, dP, dS, dV]
    
    t = np.arange(t_start, t_end, dt)
    sol = odeint(bioreactor_model, y0, t)
    
    df = pd.DataFrame(sol, columns=['X', 'P', 'S', 'V']).assign(t=t)
    df['F'] = [feeding(i) for i in t]
    return df

def generate_training_data(mu_max: float = MU_MAX, K_s: float = K_S, Y_xs: float = Y_XS, Y_px: float = Y_PX, S_f: float = S_F, samples: int = 1000):
    ''' Generate training data for the bioreactor model 
    - samples: number of samples to generate
    - inputs = F
    - outputs = X, S, P, V
    '''
    def mu(S):
        return mu_max * S / (K_s + S)

    inputs, outputs = [], []
    for _ in tqdm(range(samples)):
        F = np.random.uniform(0, 1)
        X = np.random.uniform(0, 6)
        S = np.random.uniform(0, 12)
        P = np.random.uniform(0, 2)
        V = np.random.uniform(0, 20)
        
        dX = mu(S) * X - F * X / V
        dS = -mu(S) * X / Y_xs + F * (S_f - S) / V
        dP = Y_px * mu(S) * X - F * P / V
        dV = F
        
        X_next = X + dX * dt 
        S_next = S + dS * dt    
        P_next = P + dP * dt
        V_next = V + dV * dt
        
        inputs.append([F, X, S, P, V])
        outputs.append([X_next, S_next, P_next, V_next])
        
        df = pd.DataFrame(inputs, columns=['F', 'X', 'S', 'P', 'V'])
        df['X_next'], df['S_next'], df['P_next'], df['V_next'] = np.array(outputs).T
        
    return df
    