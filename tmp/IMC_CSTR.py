import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Ορισμός σημείου ισορροπίας ---
h_eq = 30.25  # Ύψος ισορροπίας
Cb_eq = 21.48  # Συγκέντρωση ισορροπίας
w1_eq = 1  # Ροή ισορροπίας

# --- Παράμετροι του Συστήματος ---
Cb1 = 24.9  # Συγκέντρωση εισερχόμενης ροής 1
Cb2 = 0.1   # Συγκέντρωση εισερχόμενης ροής 2
k1 = 1.0    # Σταθερά αντίδρασης
k2 = 1.0    # Παράμετρος αντίδρασης
w2 = 0.1    # Σταθερή διαταραχή

# --- Υπολογισμός Γραμμικοποιημένου Μοντέλου ---
A11 = -0.1 / np.sqrt(h_eq)
A12 = 0
A21 = (-(Cb1 - Cb_eq) * w1_eq) / (h_eq ** 2) + (-(Cb2 - Cb_eq) * w2) / (h_eq ** 2)
A22 = -((w1_eq + w2) / h_eq) - (k1 - k1 * k2 * Cb_eq) / (1 + k2 * Cb_eq) ** 3

B1 = 1
B2 = (Cb1 - Cb_eq) / h_eq

C1, C2 = 0, 1  # Έξοδος είναι το Cb

# --- Δημιουργία Μοντέλου Καταστασης---
A = np.array([[A11, A12], [A21, A22]])
B = np.array([[B1], [B2]])
C = np.array([[C1, C2]])
D = np.array([[0]])

# Μετατροπή σε συνάρτηση μεταφοράς
sys_ss = ctrl.ss(A, B, C, D)
sys_tf = ctrl.ss2tf(sys_ss)
sys_tf = ctrl.minreal(sys_tf)

# --- Ρύθμιση του IMC ---
λ = 0.5 # Τιμή φίλτρου
F = ctrl.tf([1], [λ, 1])  # Φίλτρο

# Υπολογισμός του IMC Ελεγκτή
Gm_inv = ctrl.minreal(1 / sys_tf)  
Q = Gm_inv * F  

# --- Προσομοίωση του Μη Γραμμικού Συστήματος ---
def cstr_system(t, y, w1):
    """ Ορισμός του πραγματικού μη γραμμικού συστήματος """
    h, Cb = y
    dh_dt = w1 + w2 - 0.2 * np.sqrt(h)
    dCb_dt = ((Cb1 - Cb) * w1 / h + (Cb2 - Cb) * w2 / h - k1 * Cb / (1 + k2 * Cb) ** 2)
    return [dh_dt, dCb_dt]

# --- Προετοιμασία Προσομοίωσης ---
T_sim = 100  # Διάρκεια προσομοίωσης
dt = 1       # Χρονικό βήμα
time_steps = np.arange(0, T_sim, dt)

# Setpoint για το Cb
Cb_ref_values = [20.9, 21.5, 20.5]  
set_point_change_intervals = [20, 40]

# Αρχικές συνθήκες
y0 = [h_eq, 21]
Cb_ref = Cb_ref_values[0]

# Αποθήκευση αποτελεσμάτων
h_values, Cb_values, w1_values = [], [], []

# Έναρξη προσομοίωσης
for t in time_steps:
    # Ενημέρωση του setpoint αν αλλάξει ο χρόνος
    if t in set_point_change_intervals:
        Cb_ref = Cb_ref_values[set_point_change_intervals.index(t) + 1]

    # Υπολογισμός του σφάλματος
    e = Cb_ref - y0[1]


    # Υπολογισμός του σήματος ελέγχου u(t) μέσω του IMC
    t_temp, u_values = ctrl.forced_response(Q, [0, dt], e)
    u_t = u_values[-1]

    # Υπολογισμός του νέου w1
    w1 = np.clip(w1_eq + u_t, 0, None)  # Περιορισμός: δεν μπορεί να είναι < 0


    # Επίλυση του μη γραμμικού συστήματος για το επόμενο βήμα
    sol = solve_ivp(cstr_system, (t, t+dt), y0, args=(w1,), method='RK45')

    # Αποθήκευση των τιμών
    h, Cb = sol.y[:, -1]
    h_values.append(h)
    Cb_values.append(Cb)
    w1_values.append(w1)

    # Ενημέρωση αρχικών συνθηκών για το επόμενο βήμα
    y0 = [h, Cb]

# --- Οπτικοποίηση των Αποτελεσμάτων ---
Cb_ref_plot = np.zeros_like(time_steps)  # Αρχικοποίηση πίνακα για το setpoint
for i, t in enumerate(time_steps):
    if t < set_point_change_intervals[0]:
        Cb_ref_plot[i] = Cb_ref_values[0]
    elif t < set_point_change_intervals[1]:
        Cb_ref_plot[i] = Cb_ref_values[1]
    else:
        Cb_ref_plot[i] = Cb_ref_values[2]

plt.figure(figsize=(10, 5))

plt.subplot(3, 1, 1)
plt.plot(time_steps, h_values, label="h(t)", color='blue')
plt.ylabel("Ύψος h")
plt.grid()
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_steps, Cb_values, label="Cb(t)", color='orange')
plt.plot(time_steps, Cb_ref_plot, 'r--', label="Setpoint")
plt.ylabel("Συγκέντρωση Cb")
plt.grid()
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_steps, w1_values, label="w1(t)", color='green')
plt.xlabel("Χρόνος")
plt.ylabel("Ροή εισόδου w1")
plt.grid()
plt.legend()

plt.suptitle("IMC Έλεγχος του Μη Γραμμικού CSTR")
plt.show()
