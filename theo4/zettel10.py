#!python3
from matplotlib import pyplot as plt
import numpy as np
import std


m = 1

def plot_ddim(d, rel):
    eps = np.linspace(0, 100, 100000)
    g_cl = lambda e: m * ((2 * m * e) ** (0.5 * d - 1))
    g_rel = lambda e: e ** (d - 1) / 3 ** d
    if rel:
        plt.plot(eps, g_rel(eps), label=f"d={d}")
    else:
        plt.plot(eps, g_cl(eps), label=f"d={d}")


def main():
    for d in range(1, 4):
        plot_ddim(d, False)
    std.default.plt_pretty("$\\epsilon$", "G")
    plt.legend()
    plt.title("Klassisch")
    # plt.show()
    plt.savefig("clasical.pdf")
    plt.cla()
    for d in range(1, 4):
        plot_ddim(d, True)
    std.default.plt_pretty("$\\epsilon$", "G")
    plt.legend()
    plt.title("Relativistisch")
    plt.savefig("relativistic.pdf")
    # plt.show()


if __name__ == "__main__":
    main()
