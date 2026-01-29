#!python

import std
import numpy as np
from matplotlib import pyplot as plt

def fermi_func(epsilon, mu=1, beta=1):
    return 1 / (np.exp(beta * (epsilon - mu)))


def derivative_fermi_func(epsilon, mu=1, beta=1):
    return np.exp(beta * (epsilon - mu)) / (beta * np.exp(beta * epsilon - mu) + 1) ** 2


def main():
    e = np.linspace(-3, 3, 100000)    
    std.default.plt_pretty("$\\epsilon$ / a.U.", "f' / a.U.")
    plt.plot(e, derivative_fermi_func(e, mu=0, beta=3))
    plt.show()


if __name__ == "__main__":
    main()
