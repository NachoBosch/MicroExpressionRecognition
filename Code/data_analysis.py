import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir = "val_casme.csv"

df = pd.read_csv(dir)

plt.figure(figsize=(10,7))
# value_count = df.Grade.value_counts(True)
# value_count.plot(kind='bar',color=color)
color = plt.cm.viridis(np.linspace(0, 1, len(df)))
plt.bar(df['class'], df['n_images'], color=color)
plt.ylabel('Balance Dataset')
plt.show()