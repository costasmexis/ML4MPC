import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp


# Ορισμός των παραμέτρων
mu_max, Ks, Yxs, Sin = 0.2, 1, 0.5, 10.0
t_span = (0, 10)  # Χρονικό εύρος προσομοίωσης
dt = 0.1  # Χρονικό βήμα

# Ορισμός της δυναμικής του συστήματος
def system_odes(t, y, F):
    X, S, V = y
    dX_dt = (mu_max * S / (Ks + S)) * X - (F / V) * X
    dS_dt = -(1 / Yxs) * (mu_max * S / (Ks + S)) * X + (F / V) * (Sin - S)
    dV_dt = F
    return [dX_dt, dS_dt, dV_dt]



# Δημιουργία training dataset
def generate_training_data(samples=1000):
    X_data, Y_data = [], []
    for _ in range(samples):
        X0 = np.random.uniform(0.1, 5.0)  # Αρχική συγκέντρωση κυττάρων
        S0 = np.random.uniform(5, 30.0)  # Αρχική συγκέντρωση υποστρώματος
        V0 = np.random.uniform(0.5, 2.0)  # Αρχικός όγκος
        F = np.random.uniform(0.5, 2)  # Ροή εισόδου

        sol = solve_ivp(system_odes, t_span, [X0, S0, V0], args=(F,), t_eval=np.arange(t_span[0], t_span[1], dt))

        for i in range(len(sol.t) - 1):
            X_t = [sol.y[0, i], sol.y[1, i], sol.y[2, i], F]  # Features
            Y_t = [sol.y[0, i + 1], sol.y[1, i + 1], sol.y[2, i + 1]]  # Next step

            X_data.append(X_t)
            Y_data.append(Y_t)

    return np.array(X_data), np.array(Y_data)




X_train, Y_train = generate_training_data()


from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Κανονικοποίηση δεδομένων
scaler = StandardScaler()
mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', max_iter=500, solver='adam')

# Συνδυασμός scaler + MLP
model = Pipeline([("scaler", scaler), ("mlp", mlp)])

# Εκπαίδευση του MLP
model.fit(X_train, Y_train)











