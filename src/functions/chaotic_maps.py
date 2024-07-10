import numpy as np
import matplotlib.pyplot as plt

def logistic(x: float, r: float=3.99):
    return r * x * (1 - x)

def sine(x: float, a: float=1.):
    return a * np.sin(np.pi * x)

def tent(x: float, mi: float=1.99):
    return mi * np.minimum(x, 1-x)

if(__name__ == "__main__"):
    x = np.linspace(0, 4, 50000)
    y = np.zeros_like(x)
    for i in range(0, len(x)):
        num = np.random.uniform()
        for j in range(100):
            num = logistic(num, x[i])
        y[i] = num
    plt.scatter(x, y, s=1.)
    plt.xlabel("Parameter r")
    plt.ylabel("Value")
    plt.title("Logistic chaotic map")
    plt.grid(True)
    plt.show()

    """
    x = np.linspace(0, 100, 101)
    y = [0.52]
    for i in range(1, len(x)):
        num = logistic(y[i-1])
        y.append(num)
    plt.plot(x, y)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Logistic map for parameter r=3.99")
    plt.subplots_adjust(bottom=0.2)
    plt.grid(True)
    plt.show()
    """