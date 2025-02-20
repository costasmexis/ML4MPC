import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Παράμετροι του συστήματος
params = [0.2, 1, 0.5, 10.0]  # mu_max, Ks, Yxs, Sin

# Αρχικές συνθήκες
X0, S0, V0 = 0.05, 10.0, 1.0

# Absolute Time
At = 100
dt = 1
SS = int(At / dt)

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

# Κρίσιμη Ενίσχυση (Ku) και Περίοδος Pu
Ku = 2.5  # Προσδιορίζεται πειραματικά δοκιμάζοντας διάφορες τιμές μέχρι το σύστημα να ταλαντωθεί
Pu = 10.0 # Μετρούμε την περίοδο ταλάντωσης

# Υπολογισμός των PID παραμέτρων βάσει Ziegler-Nichols
Kp = 0.6 * Ku
Ki = 2 * Kp / Pu
Kd = Kp * Pu / 8

# Αρχικοποίηση PID
integral_error = 0
prev_error = 0

# PID CONTROLLER
def PID(t, X):
    global integral_error, prev_error
    err = Cb_set(t) - X
    integral_error += err * dt  # Ολοκλήρωμα
    derivative_error = (err - prev_error) / dt if t > 0 else 0  # Παράγωγος
    prev_error = err

    # Υπολογισμός PID εξόδου
    F_out = Kp * err + Ki * integral_error + Kd * derivative_error

    # Περιορισμός της παροχής μεταξύ 0 και 10 L/h
    return np.clip(F_out, 0, 10)


# Αρχικοποίηση συστήματος
X = np.zeros(SS+1)
S = np.zeros(SS+1)
V = np.zeros(SS+1)
F_PID = np.zeros(SS)

X[0], S[0], V[0] = X0, S0, V0

# PID FEEDBACK LOOP
for step in range(SS):
    t = step * dt
    F_PID[step] = PID(t, X[step])  # Το F πρέπει να είναι μη αρνητικό
    solution = solve_ivp(
        plant_model, t_span=(t, t+dt), y0=[X[step], S[step], V[step]], args=(F_PID[step],)
    )
    X[step+1], S[step+1], V[step+1] = solution.y[:, -1]

# Γράφημα αποτελεσμάτων
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

ax = axs[0]
ax.plot(range(At), [Cb_set(t) for t in range(At)], label="Setpoint", linestyle="dashed")
ax.plot(np.arange(0, At+dt, dt), X, label="Biomass Concentration")
ax.set_xlabel('Time')
ax.set_ylabel('Biomass Concentration')
ax.legend()

ax = axs[1]
ax.plot(np.arange(0, At, dt), F_PID, label="Feed Input")
ax.set_xlabel('Time')
ax.set_ylabel('Feed input')
ax.legend()

plt.show()


