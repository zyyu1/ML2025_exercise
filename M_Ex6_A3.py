import numpy as np
import matplotlib.pyplot as plt

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2
#define activation function 

def forward(x, W1, b1, W2, b2, W3, b3, act_fn):
    z1 = x.dot(W1) + b1
    a1 = act_fn(z1)
    z2 = a1.dot(W2) + b2
    a2 = act_fn(z2)
    z3 = a2.dot(W3) + b3
    return z1, a1, z2, a2, z3
#define forward propagation

def predict(x_input, W1, b1, W2, b2, W3, b3, act_fn, y_mean, y_std, x_mean, x_std):
    x_scaled = (x_input - x_mean) / x_std
    _, _, _, _, y_pred = forward(x_scaled, W1, b1, W2, b2, W3, b3, act_fn)
    return y_pred * y_std + y_mean
#define predict function 

def init_weights(act_name='relu', seed=None):
    if seed is not None:
        np.random.seed(seed)
    if act_name == 'relu':
        W1 = np.random.randn(1, 8) * np.sqrt(2 / 1)
        W2 = np.random.randn(8, 8) * np.sqrt(2 / 8)
        W3 = np.random.randn(8, 1) * np.sqrt(2 / 8)
        #he for Relu
    else:  
        W1 = np.random.randn(1, 8) * np.sqrt(1 / 1)
        W2 = np.random.randn(8, 8) * np.sqrt(1 / 8)
        W3 = np.random.randn(8, 1) * np.sqrt(1 / 8)
        # Xavier for tanh

    b1 = np.zeros((1, 8))
    b2 = np.zeros((1, 8))
    b3 = np.zeros((1, 1))
    return W1, b1, W2, b2, W3, b3
#initialize data points 


def train(act_name='relu', W1=None, b1=None, W2=None, b2=None, W3=None, b3=None, lr=0.00001, epochs=100000):

    act_fn = relu if act_name == 'relu' else tanh
    act_fn_deriv = relu_derivative if act_name == 'relu' else tanh_derivative
    #choose activation function 

    x_raw = np.linspace(-1, 1, 1000).reshape(-1, 1)
    y_true = 3*x_raw**4 - 360*x_raw**3 + 2*x_raw**2 + 5.5*x_raw + 0.5
    # Generate data

    x_mean, x_std = np.mean(x_raw), np.std(x_raw)
    y_mean, y_std = np.mean(y_true), np.std(y_true)
    x = (x_raw - x_mean) / x_std
    y = (y_true - y_mean) / y_std
    #Normalize

    if W1 is None:
        W1, b1, W2, b2, W3, b3 = init_weights(act_name)
    # Initialize weights

    losses = []

    for epoch in range(epochs):

        z1, a1, z2, a2, y_pred = forward(x, W1, b1, W2, b2, W3, b3, act_fn)
        loss = np.mean((y_pred - y)**2)
        losses.append(loss)
        # Forward

        dL_dy = 2 * (y_pred - y) / len(x)
        dW3 = a2.T @ dL_dy
        db3 = np.sum(dL_dy, axis=0, keepdims=True)

        da2 = dL_dy @ W3.T
        dz2 = da2 * act_fn_deriv(z2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * act_fn_deriv(z1)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        # Backward

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3
        # Update

    return W1, b1, W2, b2, W3, b3, losses, y_mean, y_std, x_mean, x_std
    # define train loop


#plot

x_raw = np.linspace(-1, 1, 1000).reshape(-1, 1)
y_true = 3*x_raw**4 - 360*x_raw**3 + 2*x_raw**2 + 5.5*x_raw + 0.5

W1_relu, b1_relu, W2_relu, b2_relu, W3_relu, b3_relu = init_weights('relu', seed=14)
W3_relu += np.random.normal(scale=1e-3, size=W3_relu.shape)
#Add a small perturbation 
W1_tanh, b1_tanh, W2_tanh, b2_tanh, W3_tanh, b3_tanh = init_weights('tanh', seed=6)
# Init

W1_r, b1_r, W2_r, b2_r, W3_r, b3_r, loss_relu, y_mean, y_std, x_mean, x_std = train('relu', W1_relu, b1_relu, W2_relu, b2_relu, W3_relu, b3_relu)
W1_t, b1_t, W2_t, b2_t, W3_t, b3_t, loss_tanh, _, _, _, _ = train('tanh', W1_tanh, b1_tanh, W2_tanh, b2_tanh, W3_tanh, b3_tanh)
# Train

y_pred_relu = predict(x_raw, W1_r, b1_r, W2_r, b2_r, W3_r, b3_r, relu, y_mean, y_std, x_mean, x_std)
y_pred_tanh = predict(x_raw, W1_t, b1_t, W2_t, b2_t, W3_t, b3_t, tanh, y_mean, y_std, x_mean, x_std)
# ReLU and tanh training


plt.plot(np.array(loss_relu) * 1e6, label="ReLU")
plt.plot(np.array(loss_tanh) * 1e6, label="tanh")
plt.title("Loss Convergence: ReLU vs tanh")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
# plot loss

plt.scatter(x_raw, y_true, label='Actual', s=10, marker='x', color='blue')
plt.scatter(x_raw, y_pred_relu, label='Predicted ReLU', s=10, marker='o', color='orange')
plt.scatter(x_raw, y_pred_tanh, label='Predicted tanh', s=10, marker='^', color='green')
plt.title("Approximation Function")
plt.legend(loc='upper right')
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# plot prediction function 


_, _, _, _, _, _, loss_high, _, _, _, _ = train('relu', lr=0.001)
_, _, _, _, _, _, loss_mid,  _, _, _, _ = train('relu', lr=0.0005)
_, _, _, _, _, _, loss_low,  _, _, _, _ = train('relu', lr=0.0001)
#different learning rates

plt.plot(loss_high, label='ReLU lr=0.001')
plt.plot(loss_mid,  label='ReLU lr=0.0005')
plt.plot(loss_low, label='ReLU lr=0.0001')
plt.legend()
plt.title('Learning Rate Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale("log")
plt.grid(True)
plt.show()
# Plot learning rate comparison