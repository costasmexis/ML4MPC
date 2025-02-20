import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Παράμετροι του συστήματος
params = [0.2, 1, 0.5, 10.0]  # mu_max, Ks, Yxs, Sin

# Αρχικές συνθήκες
X0 = 0.05
S0 = 10.0
V0 = 1.0
F0 = 1

# Βήμα διακριτοποίησης (για το discrete model)
h = 0.1

# MPC Step (αλλάζει δυναμικά)
dt = 10  # Μπορείς να αλλάξεις την τιμή ελεύθερα

#Absolute Time
At=100
# Simulation Steps
SS=int(At/dt)

# Prediction and control horizon
Np = 5
bnds = [(0.0, 2.0) for _ in range(Np)]

# Setpoint function
def Cb_set(t):
    if t <= 20:
        return 1.5
    elif 21 <= t < 40:
        return 3
    else:
        return 5

# Plant Model
def plant_model(t, y, F):
    X, S, V = y
    mu_max, Ks, Yxs, Sin = params
    dX_dt = (mu_max * S / (Ks + S)) * X - (F / V) * X
    dS_dt = -(1 / Yxs) * (mu_max * S / (Ks + S)) * X + (F / V) * (Sin - S)
    dV_dt = F
    return np.array([dX_dt, dS_dt, dV_dt])

# Διακριτοποιημένο Μοντέλο
def discrete_model(X, S, V, F,h):
    k1 = plant_model(t, [X, S, V], F)
    k2 = plant_model(t + h / 2, [X + k1[0] * h / 2, S + k1[1] * h / 2, V + k1[2] * h / 2], F)
    k3 = plant_model(t + h / 2, [X + k2[0] * h / 2, S + k2[1] * h / 2, V + k2[2] * h / 2], F)
    k4 = plant_model(t + h, [X + k3[0] * h, S + k3[1] * h, V + k3[2] * h], F)

    X_next = X + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    S_next = S + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    V_next = V + (h / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return X_next, S_next, V_next

# Cost Function
def cost_function(F_opt, X, S, V, t, Q=0.5, R=3):
    J = 0
    X_curr, S_curr, V_curr = X, S, V
    for k in range(Np):
        Cb = Cb_set(t + k * dt)
        X_next, S_next, V_next = discrete_model(X_curr, S_curr, V_curr, F_opt[k], h)
        J += Q * (Cb - X_next) ** 2
        if k > 0:
            J += R * (F_opt[k] - F_opt[k - 1]) ** 2
        X_curr, S_curr, V_curr = X_next, S_next, V_next
    return J

# Initialize system variables
X = np.ones(SS+1)
S = np.ones(SS+1)
V = np.ones(SS+1)
F = np.ones(SS)
X[0], S[0], V[0] = X0, S0, V0

# MPC Loop
for step in np.arange(0, SS):
    t=step*dt
    res = minimize(cost_function, F0 * np.ones(Np), args=(X[step], S[step], V[step], t), bounds=bnds, method="SLSQP")
    F[step] = res.x[0]
    sol = solve_ivp(plant_model, t_span=(t, t + dt), y0=[X[step], S[step], V[step]], args=(F[step],))
    X[step + 1], S[step + 1], V[step + 1] = sol.y[:, -1]

fig,axs=plt.subplots(1,2)
ax=axs[0]
ax.plot(range(0,At),[Cb_set(t) for t in range(0,At)])
ax.plot(np.arange(0,At+dt,dt),X)
ax.set_xlabel('Real time')
ax.set_ylabel('Biomass Concentration')
ax=axs[1]
ax.plot(np.arange(0,At,dt),F)
ax.set_xlabel('Real time')
ax.set_ylabel('Feed input')


plt.show()
