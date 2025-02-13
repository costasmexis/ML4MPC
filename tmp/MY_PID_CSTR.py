import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

# --- Parameters ---
dt = 1  # Time step for control updates
Cb1 = 24.9  # Concentration of stream 1
Cb2 = 0.1  # Concentration of stream 2
k1 = 1.0  # Reaction rate constant
k2 = 1.0  # Reaction parameter
L = 60  # Simulation time steps
h0 = 10.0  # Initial height
Cb0 = 22.0  # Initial concentration
w2 = 0.1  # Constant disturbance
Cb_ref_values = [20.9, 21.0, 20.5]  # Set point changes over time
set_point_change_intervals = [20, 40]  # Time intervals for set point changes

# PID controller parameters
Kp = 1.5
Ki = 1
Kd = 0.3

# Control input bounds
w1_min, w1_max = 0, 4.0


# --- Real System Dynamics ---
def system_of_odes(t, y, w1, w2):
    h, Cb = y
    dh_dt = w1 + w2 - 0.2 * np.sqrt(h)
    dCb_dt = ((Cb1 - Cb) * w1 / h + (Cb2 - Cb) * w2 / h - k1 * Cb / (1 + k2 * Cb) ** 2)
    return [dh_dt, dCb_dt]


# --- Set Point Function ---
def set_point(t):
    if t < set_point_change_intervals[0]:
        return Cb_ref_values[0]
    elif t < set_point_change_intervals[1]:
        return Cb_ref_values[1]
    else:
        return Cb_ref_values[2]


# --- Simulation Setup ---
Cb = np.zeros(L + 1)
Cb[0] = Cb0
w1 = np.zeros(L)
h = np.zeros(L + 1)
h[0] = h0

# Initialize PID controller variables
integral = 0.0
previous_error = 0.0

# --- Simulation Loop ---
for idx in tqdm(range(L)):
    # Get the current time
    current_time = idx * dt

    # Calculate the set point
    Cb_ref = set_point(current_time)

    # Calculate the control error
    error = Cb_ref - Cb[idx]

    # Update PID terms
    integral += error * dt
    derivative = (error - previous_error) / dt
    control_action = Kp * error + Ki * integral + Kd * derivative

    # Clamp the control action to the allowable range
    w1[idx] = np.clip(control_action, w1_min,w1_max)

    # Solve the system of ODEs for the current step
    sol = solve_ivp(
        system_of_odes,
        [idx * dt, (idx + 1) * dt],
        [h[idx], Cb[idx]],
        args=(w1[idx], w2),
        method="RK45"
    )

    # Update the state variables
    h[idx + 1], Cb[idx + 1] = sol.y[:, -1]

    # Update the previous error for the next step
    previous_error = error

# --- Plot Results ---
plt.figure(figsize=(12, 6))

# Concentration plot
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(Cb)) * dt, Cb, label="True Concentration (Cb)", color="orange")
plt.plot(np.arange(len(Cb)) * dt, [set_point(t) for t in np.arange(len(Cb)) * dt],
         label="Set Point (Cb_ref)", linestyle="--", color="red")
plt.xlabel("Time")
plt.ylabel("Concentration (Cb)")
plt.title("Concentration vs. Time (PID Controller)")
plt.legend()
plt.grid()

# Control input plot
plt.subplot(2, 1, 2)
plt.step(np.arange(len(w1)) * dt, w1, where="post", label="Control Input (w1)", color="blue")
plt.xlabel("Time")
plt.ylabel("Inflow Rate (w1)")
plt.title("Control Input vs. Time (PID Controller)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

