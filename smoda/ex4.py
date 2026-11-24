# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt


def plt_pretty(xlabel, ylabel):
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.gca().minorticks_on()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()



# %% [markdown]
# reading and plotting the data

# %%
# not motivated to do exception handling anymore
_, x = np.genfromtxt("./measurements.csv", delimiter=",", skip_header=1, unpack=True)
plt.hist(x, 15)
plt_pretty("measured value", "count")
plt.show()


#%% [markdown]
# now find $\lambda$ and $k$ numerically by maximizing the log likelyhood function for the given $x_i$.

# %%
def log_likelyhood(x, args):
    k = args[0]
    j = args[1]
    n = np.shape(x)[0]
    return n * np.log(k) - n * k * np.log(j) + np.sum((k - 1) * np.log(x)  - (x / j) ** k)


def make_neg_log_l(dataset):
    return lambda args: - log_likelyhood(dataset, args)


params = opt.minimize(make_neg_log_l(x), [1, 1])
print(f"by minization, k is {round(params.x[0], 4)} and lambda is {round(params.x[1], 4)}")


# %% [markdown]
# calculating the variance via the MC method

# %%
def gen_mc_fits(runs, length, k, j):
    mc_samples = j * ((- np.log(np.random.rand(length, runs))) ** (1 / k))
    return np.transpose(np.array([opt.minimize(make_neg_log_l(run), [1, 1]).x for run in mc_samples]))

        
mc_results = gen_mc_fits(200, 1000, *params.x)

var_k = np.var(mc_results[0])
var_lambda = np.var(mc_results[1])
print(f"the standard deviation of k is {round(np.sqrt(var_k), 4)}")
print(f"the standard deviation of lambda is {round(np.sqrt(var_lambda), 4)}")


# %% [markdown]
# calculating the bias of the two estimators via a MC method. originally estimated value is being subracted from the expectation value of the MC.
# since the bias is an order of magnitude lower then the standard deviation, both estimators appear essentially unbiased in practise

# %%

bias_k = np.average(mc_results[0]) - params.x[0]
bias_lambda = np.average(mc_results[1]) - params.x[1]

print(f"the bias of the k estimator is {round(bias_k, 4)}")
print(f"the bias of the lambda estimator is {round(bias_lambda, 4)}")


# %% [markdown]
# calculating the correlationfactor via $\rho = \frac{Cov(\lambda, k)}{\sigma_\lambda \sigma_k}$
# where $Cov(\lambda, k) = E[\hat{\lambda} \cdot \hat{k}] - E[\hat{\lambda}] \cdot E[\hat{k}]$

# %%
cov_k_lambda = np.average(mc_results[0] * mc_results[1]) - np.average(mc_results[0]) * np.average(mc_results[1])

# print(f"cov k lambda = {cov_k_lambda}")
correlation = cov_k_lambda / (np.sqrt(var_k) * np.sqrt(var_lambda))
print(f"correlation coefficient is {correlation}")


# %% [markdown]
# as such, $\lambda$ and $k$ can be said to be lightly correlated.
