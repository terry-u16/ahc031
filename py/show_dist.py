import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("input N: ", end="")
N = int(input())
TRIAL_COUNT = 10000
T = 10000
array = np.zeros(shape=(TRIAL_COUNT, N), dtype=int)

for i in range(TRIAL_COUNT):
    row = array[i, :]
    s = set()
    s.add(0)
    s.add(T)

    while len(s) < N + 1:
        s.add(random.randint(1, T - 1))

    l = list(s)
    l.sort()

    for j in range(N):
        row[j] = l[j + 1] - l[j]

    row.sort()

df = pd.DataFrame(array)

for i in range(N):
    data = df[i]
    mean = data.mean()
    std = data.std()
    rsd = data.std() / data.mean()

    print(
        f"{i:2} | Average: {mean:>5.0f} StdDev: {std:>5.0f} RSD: {rsd:>5.3f} Skewness: {data.skew():+.3f} Kurtosis: {data.kurt():+.3f}"
    )

df = pd.DataFrame(array)
ax = sns.violinplot(df)
ax.set_ylim(0, T)

plt.show()
