#python3
import numpy as np
import std
from matplotlib import pyplot as plt


def be_num(q, eta, mu):
    gi = 1
    return gi / (np.exp(q * eta - mu) - 1)


@np.vectorize
def n(q):
    N = 6
    q = int(q)
    ns = [N - 35/11, 19/11, 8/11, 4/11, 2/11, 1/11, 1/11]
    return ns[q]


x = np.arange(6)

p0 = [100, 1]
params, _ = std.fit_func(be_num, x, n(x))
print(params)

plt.scatter(x, n(x))

x = np.linspace(0, 6, 1000)
plt.plot(x, be_num(x, *params))
plt.show()
