import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

NUM_SAMPLES = 1000
MU_MAX = 0.20 # 1/h
K_S = 1 # g/l
Y_XS = 0.5 # g/g
Y_PX = 0.2 # g/g
S_F = 10 # g/l
T_START = 0
T_END = 50
IC = [0.05, 0.0, 10.0, 1.0]  # [X0, P0, S0, V0]
F_0 = 0.05

def feeding(t: float) -> float:
    return F_0 * np.exp(0.05 * t)

def simulate(mu_max: float = MU_MAX, K_s: float = K_S, Y_xs: float = Y_XS, Y_px: float = Y_PX,\
            t_start: float = T_START, t_end: float = T_END, \
            S_f: float = S_F, y0: list = IC, num_samples: int = NUM_SAMPLES) -> pd.DataFrame:
    """ Simulate the bioreactor model 
    - IC = [X0, P0, S0, V0]  # initial conditions
    """
    def mu(S):
        return mu_max * S / (K_s + S)
    
    def bioreactor_model(y, t):
        X, P, S, V = y
        dX = mu(S) * X - feeding(t) * X / V
        dS = -mu(S) * X / Y_xs + feeding(t) * (S_f - S) / V
        dP = Y_px * mu(S) * X - feeding(t) * P / V
        dV = feeding(t)
        return [dX, dP, dS, dV]
    
    t = np.linspace(t_start, t_end, num_samples)
    sol = odeint(bioreactor_model, y0, t)
    
    df = pd.DataFrame(sol, columns=['X', 'P', 'S', 'V']).assign(t=t)
    return df

def plot_simulation(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df.t, df.X, label='Biomass')
    plt.plot(df.t, df.P, label='Product')
    plt.plot(df.t, df.S, label='Substrate')
    plt.plot(df.t, df.V, label='Volume (l)')
    plt.legend()
    plt.xlabel('Time (h)')
    plt.ylabel('Concentration (g/l)')
    plt.title('Bioreactor Simulation')
    plt.show()
    