import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(42)
#set a fixed random  seed for reproducibility

x = np.linspace(-10, 10, 100)
noise = np.random.normal(0, 5, size=x.shape)
a = 2.5
b = 0.1
y = a * x + b + noise
#generate dataset and noise

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#Split the dataset into 80% of training set and 20% of testing set

def train_linear_regression(x_train, y_train, alpha, max_iter=1000):
    theta_0, theta_1 = 0, 0
    m = len(x_train)
    loss_history = []
    #Initialize θ0 and θ1,Consider learning rate of α,set maximum number of iterations of 1000
 
    for epoch in range(max_iter):
        y_pred = theta_0 + theta_1 * x_train
        error = y_pred - y_train
        grad_0 = (1/m) * np.sum(error)
        grad_1 = (1/m) * np.sum(error * x_train)
    #compute the gradients

        theta_0 -= alpha * grad_0
        theta_1 -= alpha * grad_1
    #update parameter 

        loss = (1/(2*m)) * np.sum(error**2)
        loss_history.append(loss)
    #Convergence check: record the loss at each iteration for plot purpose

    return theta_0, theta_1, loss_history

alphas = [0.01, 0.05]
results = {}

for alpha in alphas:
    theta_0, theta_1, loss_history = train_linear_regression(x_train, y_train, alpha)

    y_test_pred = theta_0 + theta_1 * x_test
    test_mse = np.mean((y_test_pred - y_test) ** 2)
    
    results[alpha] = {
        'theta_0': theta_0,
        'theta_1': theta_1,
        'loss_history': loss_history,
        'test_mse': test_mse
    }
    print(f"Test MSE (alpha={alpha}): {test_mse:.4f}")

first_alpha = alphas[0]
plt.scatter(x_train, y_train, color='blue', label='Actual(y)')
theta0 = results[first_alpha]['theta_0']
theta1 = results[first_alpha]['theta_1']
plt.plot(x_train, theta0 + theta1 * x_train, color='red', label=f'Predicted(h(θ))')
plt.xlabel("predictor variable(x)")
plt.ylabel("Real-valued response(y)")
plt.legend()
plt.grid(True)
plt.show()
#Plot regression for alpha = 0.01

plt.figure()
for alpha in alphas:
    plt.plot(results[alpha]['loss_history'], label=f'alpha={alpha}')
plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.show()
#Plot the loss versus iterations for both learning rates in the same graph and  compare their convergence.


'''
plt.figure(figsize=(10, 6))
for alpha in alphas:
    plt.plot(results[alpha]['loss_history'], label=f'alpha={alpha}', linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''


