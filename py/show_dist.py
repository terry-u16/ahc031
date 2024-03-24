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

fig, _ = plt.subplots(1, 2, figsize=(15, 6))

ax = plt.subplot(1, 2, 1)
sns.violinplot(df, ax=ax)
ax.set_ylim(0, T)

trans = array.transpose()
corrcoef = np.corrcoef(trans)

ax = plt.subplot(1, 2, 2)
annot = N <= 10
sns.heatmap(corrcoef, vmin=-1, vmax=1, annot=annot, fmt=".2f", square=True, ax=ax, cmap="coolwarm")
plt.show()
