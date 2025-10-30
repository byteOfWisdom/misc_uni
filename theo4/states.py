#!python3
import numpy as np
import scipy
from matplotlib import pyplot as plt
from sys import argv

if not len(argv) > 3:
    print("states.py [N_A] [N_B] [E]")

Na = int(argv[1])
Nb = int(argv[2])


def count(a, e):
    return scipy.special.binom(Na, a) * scipy.special.binom(Nb, e - a)


def get_count_curve(total_energy):
    energies = np.arange(0, total_energy)
    chance = [count(e, total_energy) for e in energies]
    return energies, chance


E, ohmega = get_count_curve(int(argv[3]))
plt.plot(E, ohmega)
plt.grid(which="major")
plt.grid(which="minor", linestyle=":", linewidth=0.5)
plt.gca().minorticks_on()
plt.xlabel("Energie in Teilsystem A")
plt.ylabel("Anzahl der Zustände")
plt.show()
