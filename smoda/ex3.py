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
# # task b)
# calculating $\hat{\lambda}$ as defined in the handwritten part of this task

# %%
lambda_estimate = np.sum(ks) / len(ks)
print(f"estimated lambda for sample data is {lambda_estimate}")


# print(counts)

# %% [markdown]
# # task c)
# calculating the fisher information for the data
# as well as the lower bound of variance for $\hat{\lambda}$ as given by the Cramer-Rao inequality

# %%
fisher_information = np.sum(((ks / lambda_estimate) - 1) ** 2)
print(f"fisher information is {fisher_information}")
min_variance = 1 / fisher_information
print(f"minimum variance of lambda_estimate as given by fisher information is {min_variance}")
print(f"lambda_estimate / n = {lambda_estimate / len(ks)}")

# %% [markdown]
# task d)
# bootstrap method for calculating variance of $\hat{\lambda}$
# the value for the bootstrapped variance of $\hat{\lambda}$  is similar to the analytical variance of $\hat{\lambda} / n$
# which is in turn similar to the the cramer-rao lower bound.

# %%
n_points, n_datasets = 100, 1000
samples = np.random.poisson(lambda_estimate, (n_points, n_datasets))
bootstrap_lambda = np.sum(samples, axis=0) / n_points
variance_lambda = np.var(bootstrap_lambda)
print(f"the variance of lambda_estimate via bootstrapping is {variance_lambda}")

# %% [markdown]
# task e)
# making a freqency table and plotting that together with the expected distribution
# it appears to match up quite well

# %%
eb_params = {"fmt": " ", "elinewidth": 0.75, "capsize": 5}
x = np.arange(0, max(ks) + 1)
y = 100 * (lambda_estimate ** x) * np.exp(-lambda_estimate) / scipy.special.factorial(x)
freqs, bins = np.histogram(ks, bins=max(ks) + 1, range=(0 -0.5, max(ks) + 0.5))
plt.stairs(freqs, bins, label="measured frequecies", linewidth=2, color="blue")
plt.stairs(y, bins, label="measured frequecies", linewidth=2, color="green")
plt.errorbar(np.arange(0, 11), y, yerr=np.sqrt(y), **eb_params, color="green")
plt.errorbar(np.arange(0, 11), freqs, yerr=np.sqrt(freqs), **eb_params, color="blue")
plt_pretty("count", "frequecy")
plt.show()
