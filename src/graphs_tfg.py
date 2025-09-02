import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv("metrics_log.csv")

# === Gráfica 1: MAPE por época ===
plt.figure(figsize=(8, 4))
plt.plot(df['epoch'], df['val_mape'], color='darkorange', marker='o')
plt.title('MAPE (val) por época')
plt.xlabel('Época')
plt.ylabel('MAPE (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Gráfica 2: R² por época ===
plt.figure(figsize=(8, 4))
plt.plot(df['epoch'], df['val_r2'], color='teal', marker='s')
plt.title('R² (val) por época')
plt.xlabel('Época')
plt.ylabel('R²')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Gráfica 3: Errores individuales y pérdida total ===
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train_loss'], label='Pérdida total (train_loss)', color='black', linewidth=2)
plt.plot(df['epoch'], df['val_abs_mae'], label='Abs MAE (val)', linestyle='--')
plt.plot(df['epoch'], df['val_rel_mae'], label='Rel MAE (val)', linestyle='--')
plt.plot(df['epoch'], df['val_corr_penalty'], label='Corr Penalty (val)', linestyle='--')

plt.title('Errores y pérdida total por época')
plt.xlabel('Época')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
