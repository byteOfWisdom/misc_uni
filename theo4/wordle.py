#!python3
import numpy as np
from matplotlib import pyplot as plt
from functools import reduce

data = []
with open("words5.txt") as file:
    data = list(map(lambda x: x.strip(), file.readlines()))

alphabet = list(map(chr, range(ord('a'), ord('z') + 1)))
p = [sum(map(lambda x: int(c in x), data)) / len(data) for c in alphabet]
p = np.array(p)

# total_letters = sum(map(len, data))

# words = reduce(lambda a, b: a + b, data)
# p = np.array(list(map(lambda x: words.count(x) / total_letters, alphabet)))


info_content = - np.log2(p)


u_words = sum(map(lambda x: int("u" in x), data))
halvings = - np.log2(u_words / len(data))

print("\nknowing u halves the word list", halvings, "times")
print("\n\n\n")

freq_order = np.flip(np.argsort(p))

temp = list(np.array(alphabet)[freq_order])
plt.stairs(info_content[freq_order], list(np.linspace(0, 26, 27)))
plt.xticks(np.array(range(26)) + 0.5, temp)
plt.ylabel("Informationsgehalt")
# plt.show()
plt.savefig("info_content.png", dpi=250)
plt.cla()

plt.stairs(p[freq_order], list(np.linspace(0, 26, 27)))
plt.xticks(np.array(range(26)) + 0.5, temp)
plt.ylabel("Wahrscheinlichkeit vorzukommen")
# plt.show()
plt.savefig("prob.png", dpi=250)
plt.cla()

# expected information gain:

on_hit = - np.log2(p)
on_miss = - np.log2(1 - p)

avg = p * on_hit + (1 - p) * on_miss

# for i in range(26):
# print(f"{alphabet[i]} {round(avg[i], 2)}")

with open("tbl_p.md", "w") as file:
    file.write("x  | p(x)  | I(x)  | \\<I(x)\\>\n")
    file.write("---|-------|-------|------\n")
    for a, b, c, d in zip(alphabet, p, info_content, avg):
        file.write(f"{a} | {b}  | {round(c, 4)} | {round(d, 2)}\n")


x = np.linspace(0, 1, 1000)
plt.plot(x, - x * np.log2(x) - (1 - x) * np.log2(1 - x))
# plt.show()
plt.savefig("exp_of_prob.png", dpi=250)
plt.xlabel("Wahrscheinlichkeit")
plt.ylabel("Erwarteter Informationsgehalt")
plt.cla()
