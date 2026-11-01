# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: 3.12.4
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exercise 1
#
# With this exercise we want you to get familiar with the JupyterHub setting and let you test everything out. Also we want to introduce/repeat some basic tools for data handling and plotting data, which we will need throughout the course. We will be working with Python and different libraries throughout the semester. We want to introduce some of them in this exercise (some later).
#
# A general note: You can use all functionalities different libraries (the ones we introduce but also others you might find useful to use in the exercises and are not explicitly mentioned) offer, no need to implement everything yourself. The only exception to this is, if we ask you explicitly to implement something yourself.

# %% [markdown]
# ## Programming Exercise 1.1: Data handling
#
# In this exercise we are going to look at some ways to handle data, using the libraries `pandas` and `numpy`.

# %% [markdown]
# We will start with numpy:

# %%
import numpy as np

# %% [markdown]
# If you don't know what the following lines do, ignore them, they are just to generate some data for you

# %%
N = int(1e7)
random_data_1 = np.random.normal(0, 4, N)
random_data_2 = np.random.uniform(0, 1, N)

# %% [markdown]
# #### 1.1 a) [2.5 bonus points]
#
# We want to start with some data extraction and manipulation. Perform each of the following tasks with the `numpy`-Array `random_data_1` generated above (if not stated otherwise):
#
# - Get the number of values $> 0$
# - Get the number of values $> 0.5$  and $< 1$
# - Add an offset of $+4$ to the dataset and then again, get the number of values $> 0$
# - Add the two arrays `random_data_1` and `random_data_2`, meaning, add the elements at the same index (so add the elements at index 0, add the elements at index 1,...). The resulting array should have the same length as the two random data arrays.
# - Sum up all elements in the `random_data_2` array and print the result

# %%
def n_between(dataset, lower_bound: np.float64 = 0. - np.inf, upper_bound: np.float64 = np.inf):
    return np.sum((dataset > lower_bound) & (dataset < upper_bound))

n_gt_0 = n_between(random_data_1, 0)
print(f"number of elements > 0 is {n_gt_0}")

#number between 0.5 and 1:
n_gt_half_lt_one = n_between(random_data_1, 0.5, 1)
print(f"number of elements > 0.5 and < 1 is {n_gt_half_lt_one}")

# with offset
offset_data_1 = random_data_1 + 4
n_gt_0_offset = n_between(offset_data_1, 0)
print(f"number of elements > 0 after offset is {n_gt_0_offset}")
# the summed array is:
sum_array = random_data_1 + random_data_2
# note how this assert passes because the shape did in fact not change.
assert(np.shape(sum_array) == np.shape(random_data_1))

print(f"the sum of random_data_2 is: {np.sum(random_data_2)}")



# %% [markdown]
# Now let's go on to `pandas`. This library can be used to handle tabularized data in form of data frames.

# %%
import pandas as pd

# %% [markdown]
# Again, the next cell is just to generate some data, don't worry about it.

# %%
N = 100
dataset = pd.DataFrame({
    "A": np.arange(N),
    "B": np.arange(N) + np.random.normal(0, 10, N)
})
dataset

# %% [markdown]
# #### 1.1 b) [2.5 bonus points]
#
# Do the following tasks for the `dataset` data frame.

# %% [markdown]
# Extract the column "A"

# %%
# Your solution here
col_A = dataset["A"]
col_A

# %% [markdown]
# Sum up all values in column "A"

# %%
# Your solution here
sum_of_A = dataset["A"].sum()
sum_of_A

# %% [markdown]
# Get all rows where "B" is $> 100$

# %%
# Your solution here
where_B_gt_100 = dataset[dataset["B"] > 100]
where_B_gt_100


# %% [markdown]
# Get the first 10 rows of the dataframe

# %%
# Your solution here
fist_10_rows = dataset[:10]
fist_10_rows

# %% [markdown]
# Create a new column "C" which contains the squares of the values of "B"

# %%
# Your solution here
dataset["C"] = dataset["B"] ** 2
dataset

# %% [markdown]
# ## Programming Exercise 1.2: Plotting
#
# In this exercise we want to plot some data and create some histograms. There are different plotting libraries out there and if you are familiar with one of them feel free to use it. We will focus (and also use in the discussion of the exercises) on `matplotlib`.

# %%
import matplotlib.pyplot as plt

# %% [markdown]
# #### 1.2 a) [2 bonus points]
#
# First, we want to create some scatter and line plots of data and functions.

# %% [markdown]
# Create a line plot of the function $f(t) = \sin(t)\cdot \exp(-t/10)$ for $t\in[0, 10\pi]$. _Remember to create labels for you axes_

# %%
# Your solution here
def plt_pretty(xlabel, ylabel):
    plt.grid(which="major")
    plt.grid(which="minor", linestyle=":", linewidth=0.5)
    plt.gca().minorticks_on()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


t_range = np.linspace(0, 10 * np.pi, int(1e6))
plt.plot(t_range, (lambda x: np.sin(x) * np.exp(x / 10))(t_range), label=r"$f(t) = \sin(t)\cdot \exp(-t/10)$ for $t\in[0, 10\pi]$", color="hotpink")
plt_pretty("t", "f(t)")
plt.legend()
plt.show()

# %% [markdown]
# Create a scatter plot of the column "B" ($y$ axis) vs. the column "A" ($x$ axis) of the data frame `dataset` from exercise 1.1

# %%
# Your solution here
plt.scatter(dataset["A"], dataset["B"], marker="x")
plt_pretty("A", "B")
plt.show()

# %% [markdown]
# #### 1.2 b) [1 bonus point]
#
# Now we want to create a histogram of some data from exercise 1.1 a). Histograms can be created with different tools, e.g. `np.histogram` to get the edges and counts and then plot with e.g. `plt.stairs`. Another tool would be `plt.hist` for direct counting and plotting. You can also use more advanced libraries like `hist` and for high energy physics also `mplhep`.

# %% [markdown]
# Create a histogram with 50 bins of `random_data_1`

# %%
# Your solution here
hist, edges = np.histogram(random_data_1, 50)
plt.stairs(hist, edges, label="random_data_1", color="hotpink")
plt_pretty("value", "count")
plt.legend()
plt.show()
