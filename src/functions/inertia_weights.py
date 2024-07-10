import numpy as np
import matplotlib.pyplot as plt

def constant(c: float=0.7):
    return c

def random():
    return 0.5 + np.random.rand()/2.

def linear_decreasing(t: int, t_max: int, w_min: float=0.4, w_max: float=0.9):
    return w_max - (w_max-w_min) / t_max * t

def simulated_annealing(t: int, w_min: float=0.4, w_max: float=0.9, lambda_: float=0.95):
    return w_min + (w_max-w_min) * lambda_ ** (t-1)

def natural_exponent_e1_pso(t: int, t_max: int, w_min: float=0.4, w_max: float=0.9):
    return w_min + (w_max-w_min) * np.exp(-(10*t/t_max))

def natural_exponent_e2_pso(t: int, t_max: int, w_min: float=0.4, w_max: float=0.9):
    return w_min + (w_max-w_min) * np.exp(-(4*t/t_max)**2)

def logarithm_decreasing(t: int, t_max: int, w_min: float=0.4, w_max: float=0.9, a: float=1.):
    return w_max + (w_min-w_max) * np.log10(a + (10*t)/t_max)

if(__name__ == "__main__"):
    x = np.linspace(1, 100, 100)
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = linear_decreasing(i+1, len(x))
    plt.plot(x, y)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Linear decreasing inertia weight")
    plt.subplots_adjust(bottom=0.2)
    plt.grid(True)
    plt.show()
