import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
from torch.utils.data import Subset
import pytz
import os
import joblib
import matplotlib
matplotlib.use('Agg')  # ¡IMPORTANTE! Esto evita errores de Tkinter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
#from plot_data import *

timezone = pytz.timezone("Europe/Madrid")
num_workers=0
script_dir = Path(__file__).parent
save_dir = script_dir / "model_figures"
(save_dir).mkdir(parents=True, exist_ok=True)

def print_now(text):
    now = datetime.now(timezone)
    print(f"{text} {now}]")

def plot_pred_vs_true(y_true, y_pred, epoch=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Valor real (E_eq)')
    plt.ylabel('Predicción (E_eq)')
    title = 'Pred vs True' + (f' - Epoch {epoch}' if epoch else '')
    plt.title(title)
    plt.grid(True)
    
    filepath = os.path.join(save_dir, f'epoch_{epoch:03d}_pred_vs_true_v2.png')
    plt.savefig(filepath, dpi=150)
    plt.close()

def compute_r2(pred, target):
    target_mean = torch.mean(target)
    ss_total = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    return 1 - (ss_res / (ss_total + 1e-6))

def evaluate(model, loader, device, epoch=None, normalization_params=None):
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred, _ = model(batch)  # sustituir  pred, _ = model(batch)
            preds.append(pred.to(device))
            trues.append(batch.y[:, 0].to(device))

    preds = torch.cat(preds)
    trues = torch.cat(trues)
    
    # Desescalado si se proporcionan parámetros
    if normalization_params is not None:
        max_val = normalization_params['max'].to(device)

        preds = preds * max_val
        trues = trues * max_val
    
    r2 = compute_r2(preds, trues)
    mape = torch.mean(torch.abs((preds - trues) / (trues + 1e-6)) * 100)
    
    if epoch is not None and epoch % 5 == 0:
        plot_pred_vs_true(trues.cpu().numpy(), preds.cpu().numpy(), epoch)
    
    return {'r2': r2.item(), 'mape': mape.item()}

def create_data_loaders(dataset, batch_size=32, val_ratio=0.15, test_ratio=0.15, random_state=42):
    # Primero dividimos en train/test
    train_idx, test_idx = train_test_split(
        range(len(dataset)), 
        test_size=test_ratio, 
        random_state=random_state,
        shuffle=True
    )
    
    # Luego dividimos train en train/val
    train_idx, val_idx = train_test_split(
        train_idx, 
        test_size=val_ratio, 
        random_state=random_state,
        shuffle=True
    )
    # Creamos los DataLoaders
    train_loader = DataLoader(
        Subset(dataset, train_idx), 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader 