import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.datasets import fetch_openml
# for Loading the MNIST dataset

from sklearn.model_selection import train_test_split
#for spliting the data

def load_data(): 
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist['data'], mnist['target']
    y = y.astype(int)
    X = X / 255.0
    y_encoded = np.eye(10)[y]
    return train_test_split(X, y_encoded, test_size=1/7, random_state=42)
#load mnist

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def relu(x): 
    return np.maximum(0, x)

def relu_derivative(x): 
    return (x > 0).astype(float)
#define sigmoid and relu funktions

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

def compute_accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
#define loss and accuracy

def initialize_parameters(input_size, hidden1, hidden2, output_size):
    W1 = np.random.randn(input_size, hidden1) * np.sqrt(2. / input_size)
    b1 = np.zeros((1, hidden1))
    W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2. / hidden1)
    b2 = np.zeros((1, hidden2))
    W3 = np.random.randn(hidden2, output_size) * np.sqrt(2. / hidden2)
    b3 = np.zeros((1, output_size))
    return W1, b1, W2, b2, W3, b3
#Initialize weights and biases

def train(X_train, y_train, X_test, y_test, lr, epochs):
    input_size = X_train.shape[1]
    hidden1, hidden2, output_size = 128, 64, 10
    W1, b1, W2, b2, W3, b3 = initialize_parameters(input_size, hidden1, hidden2, output_size)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        Z1 = np.dot(X_train, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = relu(Z2)
        Z3 = np.dot(A2, W3) + b3
        A3 = softmax(Z3)

        loss = cross_entropy_loss(y_train, A3)
        train_losses.append(loss)
        #forward propagation

        m = y_train.shape[0]
        dZ3 = A3 - y_train
        dW3 = np.dot(A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, W3.T)
        dZ2 = dA2 * relu_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X_train.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        #backpropagation
        
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3
        #update weights and biases

        Z1_test = np.dot(X_test, W1) + b1
        A1_test = relu(Z1_test)
        Z2_test = np.dot(A1_test, W2) + b2
        A2_test = relu(Z2_test)
        Z3_test = np.dot(A2_test, W3) + b3
        A3_test = softmax(Z3_test)

        test_loss = cross_entropy_loss(y_test, A3_test)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    test_acc = compute_accuracy(A3_test, y_test)
    return W1, b1, W2, b2, W3, b3, train_losses, test_losses, test_acc
#train loop

def show_predictions(X_test, y_test, W1, b1, W2, b2, W3, b3):
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    idx = np.random.choice(X_test.shape[0], 3, replace=False)
    for i in idx:
        img = X_test[i].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        a1 = relu(np.dot(X_test[i:i+1], W1) + b1)
        a2 = relu(np.dot(a1, W2) + b2)
        output = softmax(np.dot(a2, W3) + b3)
        plt.title(f"True Label: {np.argmax(y_test[i])}, Predicted: {np.argmax(output)}")
        plt.axis('off')
        plt.show()

from matplotlib.ticker import FuncFormatter

def plot_losses(train_losses, test_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Convergence')
    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='y')
    ax.get_yaxis().get_offset_text().set_visible(False)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plotting

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    W1, b1, W2, b2, W3, b3, train_losses, test_losses, test_acc = train(X_train, y_train, X_test, y_test, lr=0.01, epochs=5000)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    plot_losses(train_losses, test_losses)
    show_predictions(X_test, y_test, W1, b1, W2, b2, W3, b3)
# Run full pipeline
