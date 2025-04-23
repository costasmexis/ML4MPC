import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp

# Kinetic parameters

mu_max = 0.86980  # 1/h
Ks = 0.000123762  # g/l
Yxs= 0.4357     # g/g
Sin= 286          # g/l

# Αρχικές συνθήκες
X0 = 4.9
S0 = 1.673
V0 = 1.7  #L
F0 = 0 #L/h

# PID Step
dt = 0.1
``
# Absolute time
At=6
time_range=np.arange(0,At+dt,dt)
#Simulation Steps
SS=int(At/dt)

class PID:
    def __init__(self,params : list,set_point,F_bnds):
        self.Kp,self.Ki,self.Kd=params
        self.set=set_point
        self.integral=0
        self.previous=0
        self.F_min,self.F_max=F_bnds
       
    
    def update(self,X,dt):
        err=self.set-X
        self.integral +=err*dt #Ολοκλήρωμα
        der_err= (err - self.previous) / dt
        
        output = self.Kp * err + self.Ki * self.integral + self.Kd * der_err

        # Step 2: Clip output to actuator limits
        output_clipped = np.clip(output, self.F_min, self.F_max)

        # Step 3: Anti-windup - adjust integral if saturated
        if output != output_clipped:
            self.integral -= err * dt  # Revert the last integration

        self.previous = err
        return output_clipped  # Return clipped output

# Ορισμός των διαφορικών εξισώσεων που περιγράφουν το μοντέλο
def system(t, y, F):
    X, S, V = y
    S=max(S,0)
    dX_dt = (mu_max * S / (Ks + S)) * X - (F / V) * X
    dS_dt = -(1 / Yxs) * (mu_max * S / (Ks + S)) * X + (F / V) * (Sin - S)
    dV_dt = F
    return np.array([dX_dt, dS_dt, dV_dt])

params=[0.1312,0.001376,0]
F_bnds=(0,0.1)
controller=PID(params,25,F_bnds)

# Αρχικοποίηση συστήματος
X = np.zeros(SS+1)
S = np.zeros(SS+1)
V = np.zeros(SS+1)
F_PID = np.zeros(SS+1)

X[0], S[0], V[0], F_PID[0] = X0, S0, V0, F0

# PID FEEDBACK LOOP
for step in range(SS):
    t = step * dt
    F = controller.update(X[step],dt) # Το F πρέπει να είναι μη αρνητικό
    F_PID[step+1]=F
    solution = solve_ivp(
        system, t_span=(t, t+dt), y0=[X[step], S[step], V[step]], args=(F,),
        method="RK45"
    )
    X[step+1], S[step+1], V[step+1] = solution.y[:, -1]

# Γράφημα αποτελεσμάτων
fig, axs = plt.subplots(1, 2, figsize=(15, 5))


ax = axs[0]
ax.plot(time_range, X, label="Biomass Concentration")
ax.plot(time_range,[controller.set]*len(time_range))
ax.set_xlabel('Time')
ax.set_ylabel('Biomass Concentration')
ax.legend()

ax = axs[1]
ax.step(time_range, F_PID, label="Feed Input issued by PID")
ax.set_xlabel('Time')
ax.set_ylabel('Feed input')
ax.legend()

plt.show()

