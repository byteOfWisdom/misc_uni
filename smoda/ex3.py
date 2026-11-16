# %%
import numpy as np
from matplotlib import pyplot as plt
import scipy


def plt_pretty(xlabel, ylabel):
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.gca().minorticks_on()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()



# %% [markdown]
# for simplicity, assume, data is provided in ./poisson_data.csv
# if that is not the case, an insult will be printed


# %%
ks = []
try:
    ks = np.genfromtxt("./poisson_data.csv", skip_header=1, unpack=True, dtype=int)
except Exception as _:
    print("thou fool!")



# %% [markdown]
# calculating $\hat{\lambda}$ as defined in the handwritten part of this task

# %%
lambda_estimate = np.sum(ks) / len(ks)
print(f"estimated lambda for sample data is {lambda_estimate}")


# print(counts)

# %% [markdow]
# bootstrap method for calculating variance of lambda

# %%
n_points, n_datasets = 100, 1000
samples = np.random.poisson(lambda_estimate, (n_points, n_datasets))
bootstrap_lambda = np.sum(samples, axis=0) / n_points
variance_lambda = np.var(bootstrap_lambda)
print(f"the variance of lambda_hat via bootstrapping is {variance_lambda}")

# %% [markdown]
# making a freqency table and plotting that together with the expected distribution

# %%
x = np.arange(0, max(ks) + 1)
y = 100 * (lambda_estimate ** x) * np.exp(-lambda_estimate) / scipy.special.factorial(x)
freqs, bins = np.histogram(ks, bins=max(ks) + 1, range=(0 -0.5, max(ks) + 0.5))
plt.stairs(freqs, bins, label="measured frequecies")
plt.stairs(y, bins, label="measured frequecies")
plt_pretty("", "")
plt.show()
