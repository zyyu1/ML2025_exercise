import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("Dataset for Ex3Assignment 2.csv")
#load dataset

X = df[["X2 house age", "X4 number of convenience stores"]].values
y = df["Y house price of unit area"].values

X = np.hstack((np.ones((X.shape[0], 1)), X))
#add bias term

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Split the dataset into 80% of training set and 20% of testing set

def train_bgd(X, y, alpha, max_iter):
    m, n = X.shape
    theta = np.zeros(n)
    loss_history = []

    for i in range(max_iter):
        y_pred = X @ theta
        error = y_pred - y
        grad = (1 / m) * X.T @ error
        theta -= alpha * grad
        loss = (1 / (2 * m)) * np.sum(error ** 2)
        loss_history.append(loss)

    return theta, loss_history
#Implement BGD

def train_sgd(X, y, alpha, epochs):
    m, n = X.shape
    theta = np.zeros(n)
    loss_history = []

    for epoch in range(epochs):
        j = np.random.randint(0, m)
        x_i = X[j]
        y_i = y[j]
        error = (x_i @ theta) - y_i
        gradient = error * x_i
        theta -= alpha * gradient

        loss = (1 / (2 * m)) * np.sum((X @ theta - y) ** 2)
        loss_history.append(loss)

    return theta, loss_history 
#Implement SGD

theta_bgd, loss_bgd = train_bgd(X_train, y_train, alpha = 0.001, max_iter=1000)
theta_sgd, loss_sgd = train_sgd(X_train, y_train, alpha = 0.0001, epochs=100)
#Run training

y_pred_bgd = X_test @ theta_bgd
y_pred_sgd = X_test @ theta_sgd
mse_bgd = np.mean((y_pred_bgd - y_test) ** 2)
mse_sgd = np.mean((y_pred_sgd - y_test) ** 2)
#calculate loss

print(f"BGD MSE for unseen test data: {mse_bgd:.2f}")
print(f"SGD MSE for unseen test data: {mse_sgd:.2f}")

plt.figure(figsize=(10, 6))

plt.plot(range(1000), loss_bgd, label="Batch GD", color="steelblue")
sgd_extended = np.interp(np.linspace(0, 99, 1000), np.arange(100), loss_sgd)
plt.plot(range(1000), sgd_extended, label="Stochastic GD", color="orange", linestyle="--")

plt.xlabel("Iteration")
plt.ylabel("J(θ)")
plt.title("Convergence of BGD vs SGD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#Plot loss versus

X2 = df["X2 house age"].values
X4 = df["X4 number of convenience stores"].values
Y = df["Y house price of unit area"].values

x2_range = np.linspace(X2.min(), X2.max(), 50)
x4_range = np.linspace(X4.min(), X4.max(), 50)
X2_grid, X4_grid = np.meshgrid(x2_range, x4_range)
#Create grid

X2_grid_norm = (X2_grid - X2.mean()) / X2.std()
X4_grid_norm = (X4_grid - X4.mean()) / X4.std()
#normalize grid

grid_input = np.c_[np.ones(X2_grid.ravel().shape), X2_grid.ravel(), X4_grid.ravel()]
Y_pred_grid = grid_input @ theta_bgd
Y_pred_grid = Y_pred_grid.reshape(X2_grid.shape)
#Flatten grid and add bias 

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X2, X4, Y, color='blue', label='Data points')
ax.plot_surface(X2_grid, X4_grid, Y_pred_grid, alpha=0.5, color='orange')

ax.set_xlabel("X2 house age")
ax.set_ylabel("X4 number of convenience stores")
ax.set_zlabel("Y house price of unit area")

plt.tight_layout()
plt.show()
#plot 3D

learning_rates = [0.0001, 0.001]
losses_bgd = {}
losses_sgd = {}

for alpha in learning_rates:
    _, loss_b = train_bgd(X_train, y_train, alpha=alpha, max_iter=1000)
    _, loss_s = train_sgd(X_train, y_train, alpha=alpha, epochs=100)
    losses_bgd[alpha] = loss_b
    losses_sgd[alpha] = loss_s

plt.figure(figsize=(10, 6))
for alpha in learning_rates:
    plt.plot(losses_bgd[alpha], label=f"BGD alpha={alpha}", linewidth=2, linestyle='-' if alpha==0.001 else '--')
plt.title("BGD Convergence Speed for Different Learning Rates")
plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Plot BGD comparison

plt.figure(figsize=(10, 6))
for alpha in learning_rates:
    plt.plot(losses_sgd[alpha], label=f"SGD alpha={alpha}", linewidth=2, linestyle='-' if alpha==0.001 else '--')
plt.title("SGD Convergence Speed for Different Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Plot SGD comparison