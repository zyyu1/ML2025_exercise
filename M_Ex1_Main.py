import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,3,3,4])
y = np.array([2,1,3,2])
#define array

def model_A(x) :
    return x

def model_B(x) :
    return np.full_like(x,2)
#define model


x_line = np.linspace(min(x)-1, max(x)+1, 1000)
#plot x line

plt.scatter(x, y, color = 'black', label = 'Data points')
#plot data points

plt.plot((x_line), (model_A(x_line)), color = 'red', label = 'Curve_A')

plt.plot((x_line), (model_B(x_line)), color = 'green', label = 'Curve_B')
#plot curve

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.show()


def J_loss(y_pre, y):
    m = len(y)
    return np.sum((y_pre - y )**2 / (2 * m))
#define loss funktion

loss_A = J_loss(model_A(x), y)
loss_B = J_loss(model_B(x), y)

print(f"Loss for model A is : {loss_A}")
print(f"Loss for model B is : {loss_B}")
