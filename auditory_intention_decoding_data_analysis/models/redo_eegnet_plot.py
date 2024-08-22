# %% Imports
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% Base path
base_path = "reports/eegnet/all"

# %% load
dataset = f"{base_path}/history.pkl"

with open(dataset, "rb") as file:
    history = pickle.load(file)

# %% compute metrics
history_accuracies = pd.DataFrame([h["accuracy"] for h in history]).melt()
history_loss = pd.DataFrame([h["loss"] for h in history]).melt()

# %% plot
plt.figure(figsize=(7.5, 3.75))
sns.lineplot(history_accuracies, x="variable", y="value", label="accuracy")
sns.lineplot(history_loss, x="variable", y="value", label="loss")
plt.xlabel("")
plt.ylabel("")
plt.legend()
plt.savefig("training_plot_other.png")
plt.close()

# %% compute accuracy metrics
accuracies_file = f"{base_path}/accuracies.pkl"

with open(accuracies_file, "rb") as file:
    accuracies = pickle.load(file)
print(f"Mean accuracy: {np.mean(accuracies)}; STD accuracy: {np.std(accuracies)}")

# %% compute f1 metrics
f1_file = f"{base_path}/f1.pkl"

with open(f1_file, "rb") as file:
    f1 = pickle.load(file)
print(f"Mean f1: {np.mean(f1)}; STD f1: {np.std(f1)}")
