import numpy as np
import pickle

# Carica il modello surrogato
with open("kriging_model.pkl", "rb") as file:
    sm = pickle.load(file)

# Definisci i nuovi valori di input: AoA, AoS, Mach, Altitudine
# Puoi cambiare questi valori per fare previsioni con diverse condizioni
X = np.array([[9.5, 4.5, 1.20, 8500]])

# Usa il modello surrogato per fare una previsione sul coefficiente di Cl
predicted_cl = sm.predict_values(X)

# Stampa i risultati
# print(
#     f"Valori di input: Angolo di Attacco = {X[0][0]}°, AOS = {X[0][1]}°, Mach = {X[0][2]}, Altitudine = {X[0][3]} m"
# )
# print(f"Previsione del Cl: {predicted_cl[0][0]:.4f}")

print(predicted_cl)
