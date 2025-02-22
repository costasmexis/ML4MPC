import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Fed_Batch_ML import model



# Ορισμός της δυναμικής του συστήματος
def system_odes(t, y, F):
    X, S, V = y
    dX_dt = (mu_max * S / (Ks + S)) * X - (F / V) * X
    dS_dt = -(1 / Yxs) * (mu_max * S / (Ks + S)) * X + (F / V) * (Sin - S)
    dV_dt = F
    return [dX_dt, dS_dt, dV_dt]


def Cb_set(t):
    if t<=20:
        return 1.5
    elif 20< t and t<40:
        return 3
    else:
        return 5


def model_fun(X,S,V,F,model):
    Model_Input=np.array([X,S,V,F]).reshape(1,-1)
    pred=model.predict(Model_Input)
    return pred[0]

def cost(F_opt,X,S,V,t,model):
    Xcurrent,Scurrent,Vcurrent=X,S,V
    J=0
    for k in range(Np):
        Cb=Cb_set(t+k)
        X_next,S_next,V_next=model_fun(Xcurrent,Scurrent,Vcurrent,F_opt[k],model)
        J += Q * (Cb - X_next) ** 2
        if k > 0:
            J += R * (F_opt[k] - F_opt[k - 1]) ** 2
        X_curr, S_curr, V_curr = X_next, S_next, V_next
    return J

# Ορισμός των παραμέτρων
mu_max, Ks, Yxs, Sin = 0.2, 1, 0.5, 10.0

# Αρχικές συνθήκες
X0 = 0.05
S0 = 10.0
V0 = 1.0
F0 = 1

#Σχεδιαστικές Παράμετροι
Q=5
R=3
dt=1


#Absolute Time
At=100
# Simulation Steps
SS=int(At/dt)

# Prediction and control horizon
Np = 10
bnds = [(0.0, 2.0) for _ in range(Np)]

# Initialize system variables
X = np.ones(SS+1)
S = np.ones(SS+1)
V = np.ones(SS+1)
F = np.ones(SS)
X[0], S[0], V[0] = X0, S0, V0

#MPC LOOP
for step in range(0,SS):
    t=step*dt
    res=minimize(cost,F0*np.ones(Np),args=(X[step], S[step], V[step],t,model),bounds=bnds
    ,method="SLSQP")
    F[step]=res.x[0]
    sol = solve_ivp(system_odes, t_span=(t, t + dt), y0=[X[step], S[step], V[step]], args=(F[step],))
    X[step + 1], S[step + 1], V[step + 1] = sol.y[:, -1]


# Σχεδίαση Αποτελεσμάτων
plt.figure(figsize=(13, 8))
plt.subplot(2, 1, 1)
plt.plot(np.arange(0,At+dt,dt), X, label="Biomass X")
plt.plot(range(0,At), [Cb_set(t) for t in range(0,At)], "r--", label="Setpoint")
plt.xlabel("Time")
plt.ylabel("X Concentration")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(np.arange(0,At,dt), F, label="Feed Flow Rate F")
plt.xlabel("Time")
plt.ylabel("F")
plt.legend()

plt.show()