from __future__ import print_function, division
import numpy as np
from smt.utils.misc import compute_rms_error

from smt.problems import Rosenbrock
from smt.sampling_methods import LHS
from smt.surrogate_models import LS, QP, KPLS, KRG, KPLSK, GEKPLS, MGP

try:
    from smt.surrogate_models import IDW, RBF, RMTC, RMTB

    compiled_available = True
except Exception:
    compiled_available = False

try:
    import matplotlib.pyplot as plt

    plot_status = True
except Exception:
    plot_status = False


import matplotlib.pyplot as plt
from matplotlib import cm


# to ignore warning messages
import warnings

warnings.filterwarnings("ignore")

########### Initialization of the problem, construction of the training and validation points

ndim = 2
ndoe = 20  # int(10*ndim)
# Define the function
fun = Rosenbrock(ndim=ndim)

# Construction of the DOE
# in order to have the always same LHS points, random_state=1
sampling = LHS(xlimits=fun.xlimits, criterion="ese", random_state=1)
xt = sampling(ndoe)
# Compute the outputs
yt = fun(xt)

# Construction of the validation points
ntest = 200  # 500
sampling = LHS(xlimits=fun.xlimits, criterion="ese", random_state=1)
xtest = sampling(ntest)
ytest = fun(xtest)

# To visualize the DOE points
fig = plt.figure(figsize=(10, 10))
plt.scatter(xt[:, 0], xt[:, 1], marker="x", c="b", s=200, label="Training points")
plt.scatter(
    xtest[:, 0], xtest[:, 1], marker=".", c="k", s=200, label="Validation points"
)
plt.title("DOE")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
# plt.show()

########### The LS model

# # Initialization of the model
# t = LS(print_prediction=False)

# # Add the DOE
# t.set_training_values(xt, yt[:, 0])

# # Train the model
# t.train()

# # Prediction of the validation points
# y = t.predict_values(xtest)
# print("LS,  err: " + str(compute_rms_error(t, xtest, ytest)))

# # Plot prediction/true values
# if plot_status:
#     fig = plt.figure()
#     plt.plot(ytest, ytest, "-", label="$y_{true}$")
#     plt.plot(ytest, y, "r.", label=r"$\hat{y}$")

#     plt.xlabel("$y_{true}$")
#     plt.ylabel(r"$\hat{y}$")

#     plt.legend(loc="upper left")
#     plt.title("LS model: validation of the prediction model")

# plt.show()


########### The QP model

# t = QP(print_prediction=False)
# t.set_training_values(xt, yt[:, 0])

# t.train()

# # Prediction of the validation points
# y = t.predict_values(xtest)
# print("QP,  err: " + str(compute_rms_error(t, xtest, ytest)))

# # Plot prediction/true values
# if plot_status:
#     fig = plt.figure()
#     plt.plot(ytest, ytest, "-", label="$y_{true}$")
#     plt.plot(ytest, y, "r.", label=r"$\hat{y}$")

#     plt.xlabel("$y_{true}$")
#     plt.ylabel(r"$\hat{y}$")

#     plt.legend(loc="upper left")
#     plt.title("QP model: validation of the prediction model")

# plt.show()

########### The Kriging model

# The variable 'theta0' is a list of length ndim.
t = KRG(theta0=[1e-2] * ndim, print_prediction=True)
t.set_training_values(xt, yt[:, 0])

t.train()

# Prediction of the validation points
y = t.predict_values(xtest)
print("Kriging,  err: " + str(compute_rms_error(t, xtest, ytest)))
if plot_status:
    # Plot the function and the prediction
    fig = plt.figure()
    plt.plot(ytest, ytest, "-", label="$y_{true}$")
    plt.plot(ytest, y, "r.", label=r"$\hat{y}$")

    plt.xlabel("$y_{true}$")
    plt.ylabel(r"$\hat{y}$")

    plt.legend(loc="upper left")
    plt.title("Kriging model: validation of the prediction model")

if plot_status:
    plt.show()

# Value of theta
print("theta values", t.optimal_theta)
