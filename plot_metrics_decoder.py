import pandas as pd
import matplotlib.pyplot as plt

# Cargar el CSV
df = pd.read_csv("decoder_metrics_300_V2.csv")

# --- MSE Loss ---
plt.figure(figsize=(6, 4))
plt.plot(df["epoch"], df["mse_loss"], marker='o', color='tab:blue')
plt.title("MSE Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("mse_loss_300.png", dpi=300)
plt.close()

# --- MAE ---
plt.figure(figsize=(6, 4))
plt.plot(df["epoch"], df["mae"], marker='s', color='tab:orange')
plt.title("MAE over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.grid(True)
plt.tight_layout()
plt.savefig("mae_300.png", dpi=300)
plt.close()

# --- Accuracy ---
plt.figure(figsize=(6, 4))
plt.plot(df["epoch"], df["accuracy_percent"], marker='^', color='tab:green')
plt.title("Accuracy (<2%) over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy [%]")
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_300.png", dpi=300)
plt.close()