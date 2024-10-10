import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

california = fetch_california_housing(as_frame=True)
cal = california.frame
X, y = cal[["HouseAge"]], cal["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20
)

# Modello 1: Regressione Lineare
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_y_pred = lin_model.predict(X_test)
lin_mse = mean_squared_error(y_test, lin_y_pred)
lin_r2 = r2_score(y_test, lin_y_pred)

# Modello 2: Support Vector Regression (SVR)
svr_model = SVR(kernel="rbf")
svr_model.fit(X_train, y_train)
svr_y_pred = svr_model.predict(X_test)
svr_mse = mean_squared_error(y_test, svr_y_pred)
svr_r2 = r2_score(y_test, svr_y_pred)

# Modello 3: Random Forest Regressor
# rf_model = RandomForestRegressor(n_estimators=100, random_state=20)
# rf_model.fit(X_train, y_train)
# rf_y_pred = rf_model.predict(X_test)
# rf_mse = mean_squared_error(y_test, rf_y_pred)
# rf_r2 = r2_score(y_test, rf_y_pred)

# Modello 4: Gaussian Process Regression
# kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
# gpr_model.fit(X_train, y_train)
# gpr_y_pred, gpr_std = gpr_model.predict(X_test, return_std=True)
# gpr_mse = mean_squared_error(y_test, gpr_y_pred)
# gpr_r2 = r2_score(y_test, gpr_y_pred)

# Stampa i risultati
print("Regressione Lineare:")
print(f"Mean Squared Error: {lin_mse}, R²: {lin_r2}\n")

print("Support Vector Regression:")
print(f"Mean Squared Error: {svr_mse}, R²: {svr_r2}\n")

# print("Random Forest Regressor:")
# print(f"Mean Squared Error: {rf_mse}, R²: {rf_r2}\n")

# print("Gaussian Process Regression:")
# print(f"Mean Squared Error: {gpr_mse}, R²: {gpr_r2}\n")

# Plot dei risultati
plt.figure(figsize=(14, 7))

# Scatter plot dei dati originali
plt.scatter(X, y, color="red", label="Dati Originali", alpha=0.5)

# Linea di regressione Lineare
plt.plot(
    X_test, lin_y_pred, color="blue", linewidth=2, label="Linea di Regressione Lineare"
)

# Linea di regressione SVR
plt.scatter(X_test, svr_y_pred, color="green", label="SVR Predizioni", alpha=0.5)

# # Linea di regressione Random Forest
# plt.scatter(
#     X_test, rf_y_pred, color="orange", label="Random Forest Predizioni", alpha=0.5
# )

# # Linea di regressione Gaussian Process
# plt.scatter(X_test, gpr_y_pred, color="purple", label="GPR Predizioni", alpha=0.5)

plt.title("Confronto tra Modelli di Regressione sul dataset California Housing")
plt.xlabel("House Age")
plt.ylabel("House Value")
plt.legend()
plt.show()
