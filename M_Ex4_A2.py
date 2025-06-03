import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
#Dataset

df = pd.read_csv("Dataset for Ex4Assignment 2.csv")
X = df[["X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station", "X4 number of convenience stores", "X5 latitude", "X6 longitude"]].values

y = df["Y house price of unit area"].values
y = np.where(y <= y.mean(), 1, 0)
#import data and data labeling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
n = X_scaled.shape[1]
#normalize feature and add bias

indices = np.arange(X_scaled.shape[0])       
np.random.seed(42)  
np.random.shuffle(indices) 

X_scaled = X_scaled[indices]
y = y[indices]

split_idx = int(0.8 * len(X_scaled))

X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
#Dataset Splitting

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    return -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

def compute_accuracy(y_true, y_pred_probs):
    y_pred = (y_pred_probs >= 0.5).astype(int)
    return np.mean(y_true == y_pred)

def train_logistic_regression(X, y, alpha, num_iter):
    m, n = X.shape
    theta = np.zeros(n)
    loss_history = []
    theta_history = []

    for _ in range(num_iter):
        h = sigmoid(X @ theta)
        grad = (1/m) * (X.T @ (h - y))
        theta -= alpha * grad
        loss = compute_loss(X, y, theta)
        loss_history.append(loss)
        theta_history.append(theta.copy())

    return theta, loss_history, np.array(theta_history)
#logistic regression Model Implementation

theta = np.zeros(n)
alphas = [0.001, 0.01, 0.1]
num_iter = 50000
results = {}
#Initialize

for alpha in alphas:
    theta, loss_history, theta_history = train_logistic_regression(X_train, y_train, alpha=alpha, num_iter=num_iter)
    train_acc = compute_accuracy(y_train, sigmoid(X_train @ theta))
    test_acc = compute_accuracy(y_test, sigmoid(X_test @ theta))

    results[alpha] = {
        "theta": theta,
        "loss": loss_history,
        "theta_history": theta_history,
        "train_acc": train_acc,
        "test_acc": test_acc
    }
    print(f"Alpha={alpha} → Train Accuracy: {train_acc:.3f}, Test Accuracy: {test_acc:.3f}")
    # Evaluate for multiple learning rates

if abs(train_acc - test_acc) > 0.05:
    print("overfitting detected")
else :  print("not overfitting")
#conclude if overfitting or not

for alpha in alphas:
    train_loss = results[alpha]["loss"][:1000]
    test_loss = []
    for i in range(1000):
        theta_i = results[alpha]["theta_history"][i]
        test_loss.append(compute_loss(X_test, y_test, theta_i))

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Training Loss", color='blue')
    plt.plot(test_loss, linestyle="--", label="Testing Loss", color='green')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Iterations (alpha={alpha})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

colors = ['blue', 'green', 'orange']
plt.figure(figsize=(10, 6))
for idx, alpha in enumerate(alphas):
    train_loss = results[alpha]["loss"][:1000]
    test_loss = []
    for i in range(1000):
        theta_i = results[alpha]["theta_history"][i]
        test_loss.append(compute_loss(X_test, y_test, theta_i))

    plt.plot(train_loss, color=colors[idx], label=f"Train α={alpha}", linestyle='-')
    plt.plot(test_loss, color=colors[idx], label=f"Test α={alpha}", linestyle='--')

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss vs Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
# Plot training loss

alpha_to_plot = 0.01
theta_history = results[alpha_to_plot]["theta_history"]
theta_0_vals = theta_history[:50000, 0]

plt.figure(figsize=(9, 5))
plt.plot(theta_0_vals, linestyle="--", color="green", label=r'$\theta_0$ convergence')
plt.xlabel("Iterations")
plt.ylabel(r"$\theta_0$")
plt.title("Theta Convergence vs Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
# Plot convergence of theta parameters (for each alpha)

