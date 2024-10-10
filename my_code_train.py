import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import smt.surrogate_models as sms
from smt.sampling_methods import LHS
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS, MGP
import pickle
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.utils import resample
from smt.utils.misc import compute_rms_error

# carica il dataset
data = pd.read_csv("aeromap_data.csv")


# Aumenta il numero di dati applicando una leggera perturbazione ai dati originali
def augment_data(data, num_new_samples):
    augmented_data = []
    for i in range(num_new_samples):
        # Applica una leggera variazione casuale ai dati originali
        perturbation = np.random.normal(
            0, 0.01, data.shape
        )  # Perturbazione con media 0 e deviazione standard 0.01
        new_sample = data + perturbation
        augmented_data.append(new_sample)

    return np.vstack(augmented_data)


# Aumenta il dataset
n_augment_samples = len(data) * 1  # Aumenta di 10 volte il numero di punti originali
X = data[["Angle_of_Attack", "Angle_of_Sideslip", "Mach_Number", "Altitude"]].values
y = data[["Cl", "Cd", "Cs", "Cml", "Cmd", "Cms"]].values

X_augmented = augment_data(X, n_augment_samples // len(X))
y_augmented = augment_data(y, n_augment_samples // len(y))

# Combina i dati originali e quelli aumentati
X_combined = np.vstack([X, X_augmented])
y_combined = np.vstack([y, y_augmented])

# Crea un DataFrame per la rimozione dei duplicati
data_combined = pd.DataFrame(
    np.hstack([X_combined, y_combined]),
    columns=[
        "Angle_of_Attack",
        "Angle_of_Sideslip",
        "Mach_Number",
        "Altitude",
        "Cl",
        "Cd",
        "Cs",
        "Cml",
        "Cmd",
        "Cms",
    ],
)

# Rimuovi eventuali duplicati
data_cleaned = data_combined.drop_duplicates()

print(data_cleaned)

# Dividi di nuovo X e y dopo la pulizia
X_cleaned = data_cleaned[
    ["Angle_of_Attack", "Angle_of_Sideslip", "Mach_Number", "Altitude"]
].values
y_cleaned = data_cleaned[["Cl", "Cd", "Cs", "Cml", "Cmd", "Cms"]].values

# # Estrai l'angolo di attacco e i parametri di output
# angle_of_attack = data["Angle_of_Attack"]
# outputs = ["Cl", "Cd", "Cs", "Cml", "Cmd", "Cms"]

# # Crea lo scatter plot
# plt.figure(figsize=(12, 8))

# for i, output in enumerate(outputs):
#     plt.subplot(2, 3, i + 1)  # 2 righe, 3 colonne
#     plt.scatter(angle_of_attack, data[output], alpha=0.5)
#     plt.title(f"{output} vs Angle of Attack")
#     plt.xlabel("Angle of Attack (degrees)")
#     plt.ylabel(output)

# plt.tight_layout()
# plt.show()

# Crea gli istogrammi per ciascuna variabile
# num_features = data.shape[1]
# fig, axes = plt.subplots(1, num_features, figsize=(14, 3))

# # Traccia gli istogrammi per ogni colonna del DataFrame
# for i, col in enumerate(data.columns):
#     data[col].hist(ax=axes[i], bins=10)
#     axes[i].set_title(col)

# plt.tight_layout()
# plt.show()

# data_subset = data.iloc[:,]
# fig, ax = plt.subplots()
# sns.boxplot(data=data_subset, ax=ax)
# sns.stripplot(data=data_subset, color="black", size=4, jitter=True, ax=ax)
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y_cleaned, test_size=0.3, random_state=42
)
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test, test_size=1 / 3, random_state=42
)

# print("Dimensioni di X_test:", X_test.shape)
# print("Dimensioni di y_test:", y_test.shape)
# print("Dimensioni di y_pred:", y_pred.shape)

# Assumiamo che 'ndim' sia già definito come la dimensione di X_train.
ndim = X_train.shape[1]

# Modello Kriging con diversi kernel
# t1 = KRG(theta0=[1e-2] * ndim, print_prediction=False, corr="squar_exp")
# t1.set_training_values(X_train, y_train[:, 0])
# t1.train()

# t2 = KRG(theta0=[1e-2] * ndim, print_prediction=False, corr="abs_exp")
# t2.set_training_values(X_train, y_train[:, 0])
# t2.train()

# Variamo i termini di regressione
# t31 = KRG(
#     theta0=[1e-2] * ndim, print_prediction=False, corr="matern32", poly="constant"
# ) ===
# t31.set_training_values(X_train, y_train[:, 0]) ====
# t31.train() ====

# t32 = KRG(theta0=[1e-2] * ndim, print_prediction=False, corr="matern32", poly="linear")
# t32.set_training_values(X_train, y_train[:, 0])
# t32.train()

# t33 = KRG(
#     theta0=[1e-2] * ndim, print_prediction=False, corr="matern32", poly="quadratic"
# )
# t33.set_training_values(X_train, y_train[:, 0])
# t33.train()

# t4 = KRG(theta0=[1e-2] * ndim, print_prediction=False, corr="matern52")
# t4.set_training_values(X_train, y_train[:, 0])
# t4.train()

# Previsioni sui dati di test
# y1 = t1.predict_values(X_test)
# y2 = t2.predict_values(X_test)
# y31 = t31.predict_values(X_test) =====
# y32 = t32.predict_values(X_test)
# y33 = t33.predict_values(X_test)
# y4 = t4.predict_values(X_test)

# Calcola e stampa gli errori per ciascun output
# for i in range(y_train.shape[1]):
#     print(f"\nOutput {i+1}:")
# print(
#     "Kriging squared exponential,  err: "
#     + str(compute_rms_error(t1, X_test, y_test[:, i]))
# )
# print(
#     "Kriging absolute exponential,  err: "
#     + str(compute_rms_error(t2, X_test, y_test[:, i]))
# )
# print(
#     "Kriging matern32,  err: " + str(compute_rms_error(t31, X_test, y_test[:, i]))
# )
# print(
#     "Kriging matern32,  err: " + str(compute_rms_error(t32, X_test, y_test[:, i]))
# )
# print(
#     "Kriging matern32,  err: " + str(compute_rms_error(t33, X_test, y_test[:, i]))
# )
# print("Kriging matern52,  err: " + str(compute_rms_error(t4, X_test, y_test[:, i])))

# Se desideri, puoi anche utilizzare y1, y2, y3, y4 successivamente nel codice

# print("\n")
# print("Comparison of errors")
# print(
#     "Kriging squared exponential,  err: "
#     + str(compute_rms_error(t1, X_test, y_test[:, 0]))
# )
# print(
#     "Kriging absolute exponential,  err: "
#     + str(compute_rms_error(t2, X_test, y_test[:, 0]))
# )
# print("Kriging matern32,  err: " + str(compute_rms_error(t3, X_test, y_test[:, 0])))
# print("Kriging matern52,  err: " + str(compute_rms_error(t4, X_test, y_test[:, 0])))


# def hyperparameter_tuning(X_train, y_train, X_val, y_val):
#     best_rms = float("inf")
#     best_model = None
#     best_params = None

#     # Prova diverse configurazioni di iperparametri
#     for theta in [[1e-2], [1e-3], [1e-1]]:
#         for corr in [
#             "squar_exp",
#             "abs_exp",
#             "matern32",
#             "matern52",
#         ]:
#             for poly in ["constant", "linear", "quadratic"]:
#                 # Crea il modello Kriging
#                 model = KRG(
#                     theta0=theta * ndim, print_prediction=False, corr=corr, poly=poly
#                 )

#                 # Addestra il modello
#                 model.set_training_values(X_train, y_train[:, 0])
#                 model.train()

#                 # Calcola l'errore RMS sul set di validazione
#                 rms_val = compute_rms_error(model, X_val, y_val[:, 0])

#                 # Se l'errore è migliore, aggiorna il miglior modello
#                 if rms_val < best_rms:
#                     best_rms = rms_val
#                     best_model = model
#                     best_params = {"theta": theta, "corr": corr, "poly": poly}

#     return best_model, best_params


# Esegui il tuning degli iperparametri
# best_model, best_params = hyperparameter_tuning(X_train, y_train, X_val, y_val)

# Valuta il miglior modello sul set di test
# y_test_pred = best_model.predict_values(X_test)
# rms_test = compute_rms_error(best_model, X_test, y_test[:, 0])
# print("Miglior modello con parametri:", best_params)
# print("Errore RMS sul set di test:", rms_test)

# print("Kriging matern32,  err: " + str(compute_rms_error(t31, X_test, y_test[:, 0])))


t = KRG(
    theta0=[0.0001] * ndim, print_prediction=False, corr="matern52", poly="constant"
)
# t = KPLSK(n_comp=2, theta0=[1e-2, 1e-2], print_prediction=False)

# Addestramento del modello per ogni output
for i in range(y_train.shape[1]):
    t.set_training_values(X_train, y_train[:, i])
    t.train()
    y = t.predict_values(X_test)
    rms_test = compute_rms_error(t, X_test, y_test[:, i])
    print(f"Errore RMS sul set di test per output {i}: {rms_test}")

    # Confronto tra y_test e y_test_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[:, i], y, label="Predizioni vs. Reali", alpha=0.7)
    plt.plot(
        [y_test[:, i].min(), y_test[:, i].max()],
        [y_test[:, i].min(), y_test[:, i].max()],
        "r--",
        label="y = x",
    )
    plt.xlabel("Valori Reali")
    plt.ylabel("Valori Predetti")
    plt.title("Confronto tra Valori Predetti e Reali per Cl")
    plt.legend()
    plt.grid()
plt.show()

# Stampa y e ytest per un connfronto dal terminale

print(y, y_test[:, 5])

# Salva il modeello in un file
with open("kriging_model.pkl", "wb") as file:
    pickle.dump(t, file)

print("Modello Kriging salvato con successo!")
