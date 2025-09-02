import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from processing_data import LatticeDataset, custom_collate_fn
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.data import Batch, Data
from torch.utils.data import Subset
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LatticeData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ref_coords':
            return 0
        return super().__inc__(key, value, *args, **kwargs)

class EqPredictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z):
        return self.fc(z)

class RobustEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, edge_attr_dim=4):
        super().__init__()
        self.conv1 = GATConv(in_channels=input_dim,out_channels=hidden_dim,heads=3,edge_dim=edge_attr_dim,  dropout=0.1)
        self.norm1 = nn.BatchNorm1d(hidden_dim * 3)
        self.conv2 = GATConv(in_channels=hidden_dim * 3,out_channels=hidden_dim,edge_dim=edge_attr_dim, dropout=0.1)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.mu_net = nn.Linear(hidden_dim, latent_dim)
        self.logvar_net = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        h1 = self.conv1(x, edge_index, edge_attr=edge_attr)  
        h1 = self.norm1(F.elu(h1))
        h2 = self.conv2(h1, edge_index, edge_attr=edge_attr)  
        h2 = self.norm2(F.elu(h2))

        attn_weights = self.attention(h2)
        h2 = h2 * attn_weights
        
        # Pooling global
        h_global = global_add_pool(h2, batch)
        
        return self.mu_net(h_global), self.logvar_net(h_global)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class ImprovedDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, z, batch, ref_coords):
        z_expanded = torch.cat([
            z[i].repeat((batch == i).sum(), 1) for i in range(z.size(0))
        ], dim=0)
        input_feats = torch.cat([z_expanded, ref_coords], dim=1)
        return self.mlp(input_feats)

def save_metrics_history(metrics_history, filename="metrics_per_epoch.csv"):
    df = pd.DataFrame(metrics_history)
    df.to_csv(filename, index=False)
    print(f"Guardado CSV de métricas: {filename}")

def plot_ratio_trends(metrics_history):
    df = pd.DataFrame(metrics_history)
    plt.figure(figsize=(12, 5))
    
    # Trend lines
    plt.subplot(121)
    plt.plot(df['epoch'], df['median_ratio'], label='Mediana', marker='o')
    plt.fill_between(df['epoch'], df['median_ratio']-df['std_ratio'], 
                     df['median_ratio']+df['std_ratio'], alpha=0.2)
    plt.axhline(1.0, color='red', linestyle='--')
    plt.title('Evolución del Ratio Mediano ±1σ')
    plt.legend()
    
    # Precision bars
    plt.subplot(122)
    plt.bar(df['epoch']-0.2, df['within_5%'], width=0.4, label='±5%')
    plt.bar(df['epoch']+0.2, df['within_10%'], width=0.4, label='±10%')
    plt.title('Porcentaje de Predicciones dentro de Umbrales')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ratio_metrics_trend.png')
    plt.close()
    
def residual_vs_density_plotter(E_eq_pred, E_eq_real, densities, epoch=None):

    pred = E_eq_pred.detach().cpu().numpy().flatten()
    real = E_eq_real.detach().cpu().numpy().flatten()
    densities_np = densities.detach().cpu().numpy().flatten()
    
    residuals = pred - real  # Residuos brutos (pred - real)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(densities_np, residuals, alpha=0.6, c='purple')
    plt.axhline(y=0, color='r', linestyle='--', label='Error cero')
    
    # Ajustar límites para mejor visualización
    residual_range = max(abs(residuals.min()), abs(residuals.max()))
    plt.ylim(-residual_range * 1.1, residual_range * 1.1)
    
    plt.xlabel('Densidad relativa')
    plt.ylabel('Residuo (Predicción - Real)')
    plt.title('Residuos vs Densidad Relativa')
    plt.legend()
    plt.grid(True)
    
    if epoch is not None:
        plt.savefig(f'residual_vs_density_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('residual_vs_density_final.png', dpi=300, bbox_inches='tight')
    plt.close()

def adaptive_loss(pred, target, min_val=1e-6, error_threshold=0.03, high_weight=2.0):
    denom = torch.abs(target).clamp(min=min_val)
    relative_error = torch.abs((pred - target) / denom)
    percent_error = relative_error * 100
    
    # Suavizar la transición de pesos
    weight = 1.0 + (high_weight - 1.0) * torch.sigmoid((percent_error - error_threshold)/5.0)
    return torch.mean(weight * (relative_error ** 2)), relative_error.detach()

def weighted_mse_loss(recon_nodes, real_nodes, batch, corner_weight=0.1):
    is_corner = torch.zeros_like(batch, dtype=torch.bool)
    for i in range(batch.max() + 1):
        is_corner[(batch == i).nonzero()[:8]] = True
    loss = F.mse_loss(recon_nodes[~is_corner], real_nodes[~is_corner])
    loss += corner_weight * F.mse_loss(recon_nodes[is_corner], real_nodes[is_corner])
    return loss

def identify_hard_samples(encoder, predictor, dataset):
    error_values = []
    for idx in range(len(dataset)):
        data = dataset[idx]
        with torch.no_grad():
            batch_vec = torch.zeros(data.x.size(0), dtype=torch.long)
            mu, _ = encoder(data.x, data.edge_index, data.edge_attr, batch_vec)
            pred = predictor(mu).item()
            real = data.y[0][0].item()
            error_values.append(abs(pred - real))
        
    # Convertir a pesos (muestras con mayor error tienen mayor probabilidad)
    weights = torch.tensor(error_values, dtype=torch.float) + 1e-6
    return weights / weights.sum()

################################################################################################################
#                       R^2 definition
################################################################################################################
def compute_r2(pred, target):

    # Cálculo del R²
    target_mean = torch.mean(target)
    ss_total = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    
    # Cálculo del cociente pred/real (evitando división por cero)
    epsilon = 1e-6  # Pequeño valor para estabilidad numérica
    ratios = pred / (target + epsilon)
    
    return r2.item(), ratios

def visualize_complete_latent_space(encoder, dataset,num_epoch, latent_dim, epoch=None, sample_indices=None):
    """Visualización completa del espacio latente para cada elemento individual"""
    encoder.eval()
    num_samples = dataset.len()
    # Preparar DataLoader sin shuffling para mantener correspondencia
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    
    # Recolectar todos los datos
    all_latent = []
    all_E_eq = []
    all_density = []
    all_batch_indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            mu, _ = encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            all_latent.append(mu.cpu().numpy())
            all_E_eq.append(batch.y[:, 0].cpu().numpy())
            all_density.append(batch.y[:, 1].cpu().numpy())
            all_batch_indices.extend([batch_idx]*len(mu))
    
    # Concatenar todos los resultados
    latent_vectors = np.concatenate(all_latent, axis=0)
    E_eq_values = np.concatenate(all_E_eq, axis=0)
    density_values = np.concatenate(all_density, axis=0)
    
    # Reducción de dimensionalidad con UMAP
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(latent_vectors)
    
    # Crear DataFrame para facilitar el análisis
    df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'E_eq': E_eq_values,
        'Density': density_values,
        'Batch': all_batch_indices,
        'Index': range(len(embedding))
    })
    
    # Visualización 1: Por E_eq
    plt.figure(figsize=(14, 6))
    
    plt.subplot(121)
    sc = plt.scatter(df['UMAP1'], df['UMAP2'], c=df['E_eq'], 
                    cmap='viridis', alpha=0.8, s=50)
    plt.colorbar(sc, label='E_eq (normalizado)')
    plt.title(f'Espacio Latente por E_eq\n{len(df)} muestras')
    
    # Visualización 2: Por densidad
    plt.subplot(122)
    sc = plt.scatter(df['UMAP1'], df['UMAP2'], c=df['Density'], 
                    cmap='plasma', alpha=0.8, s=50)
    plt.colorbar(sc, label='Densidad Relativa')
    plt.title('Espacio Latente por Densidad')
    
    # Resaltar muestras específicas si se indican
    if sample_indices is not None:
        for idx in sample_indices:
            if idx < len(df):
                plt.subplot(121)
                plt.scatter(df.iloc[idx]['UMAP1'], df.iloc[idx]['UMAP2'], 
                           c='red', s=150, edgecolors='black', linewidth=1.5)
                plt.subplot(122)
                plt.scatter(df.iloc[idx]['UMAP1'], df.iloc[idx]['UMAP2'], 
                           c='red', s=150, edgecolors='black', linewidth=1.5)

    plt.tight_layout()

    # Guardar o mostrar
    if epoch is not None:
        plt.savefig(f'latent_space_complete_{num_samples}_{num_epoch}_{latent_dim}_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'latent_space_complete_{num_samples}_{num_epoch}_{latent_dim}_final.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Guardar los datos para análisis posterior
    df.to_csv('latent_space_data.csv', index=False)

    return df

def compute_ratio_metrics(pred, target):
    epsilon = 1e-6
    ratios = pred / (target.abs() + epsilon)  # Usamos valor absoluto para evitar divisiones por cero
    
    metrics = {
        'mean_ratio': torch.mean(ratios).item(),
        'median_ratio': torch.median(ratios).item(),
        'std_ratio': torch.std(ratios).item(),
        'within_5%': ((ratios >= 0.95) & (ratios <= 1.05)).float().mean().item() * 100,
        'within_10%': ((ratios >= 0.90) & (ratios <= 1.10)).float().mean().item() * 100,
        'max_overpred': torch.max(ratios).item(),
        'max_underpred': torch.min(ratios).item(),
    }
    return metrics
    

def prediction_plotter(E_eq_pred, E_eq_real, epoch=None):
    """
    Gráfico combinado: residuos cuadráticos (arriba) y cocientes pred/real (abajo).
    """
    pred = E_eq_pred.detach().cpu().numpy().flatten()
    real = E_eq_real.detach().cpu().numpy().flatten()
    ratios = pred / real  # Cocientes directos (no al cuadrado)
    
    plt.figure(figsize=(12, 10))
    
    # --- Gráfico 1: Residuos cuadráticos ---
    plt.subplot(2, 1, 1)
    plt.scatter(real, ratios**2, alpha=0.6, label='Residuos cuadráticos')
    plt.axhline(y=1, color='r', linestyle='--', label='Predicción perfecta')
    plt.axhline(y=1.1**2, color='orange', linestyle=':', label='±10% error')
    plt.axhline(y=0.9**2, color='orange', linestyle=':')
    plt.ylabel('(Predicción / Real)$^2$')
    plt.title('Residuos Cuadráticos y Distribución de Cocientes')
    plt.legend()
    plt.grid(True)
    
    # --- Gráfico 2: Histograma de cocientes ---
    plt.subplot(2, 1, 2)
    plt.hist(ratios, bins=30, alpha=0.7, edgecolor='black', color='green')
    plt.axvline(x=1.0, color='r', linestyle='--', label='Ratio ideal (1.0)')
    plt.xlabel('Predicción / Real')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if epoch is not None:
        plt.savefig(f'enhanced_residuals_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('enhanced_residuals_final.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ratio_trends(metrics_history):
    df = pd.DataFrame(metrics_history)
    plt.figure(figsize=(12, 5))
    
    # Trend lines
    plt.subplot(121)
    plt.plot(df['epoch'], df['median_ratio'], label='Mediana', marker='o')
    plt.fill_between(df['epoch'], df['median_ratio']-df['std_ratio'], 
                     df['median_ratio']+df['std_ratio'], alpha=0.2)
    plt.axhline(1.0, color='red', linestyle='--')
    plt.title('Evolución del Ratio Mediano ±1σ')
    plt.legend()
    
    # Precision bars
    plt.subplot(122)
    plt.bar(df['epoch']-0.2, df['within_5%'], width=0.4, label='±5%')
    plt.bar(df['epoch']+0.2, df['within_10%'], width=0.4, label='±10%')
    plt.title('Porcentaje de Predicciones dentro de Umbrales')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ratio_metrics_trend.png')
    plt.close()

def scatter_pred_vs_real(pred, real, epoch=None):
    pred_np = pred.detach().cpu().numpy().flatten()
    real_np = real.detach().cpu().numpy().flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(real_np, pred_np, alpha=0.6, c='blue')
    plt.plot([real_np.min(), real_np.max()], [real_np.min(), real_np.max()], 'r--')
    plt.xlabel('E_eq Real (GPa)')
    plt.ylabel('E_eq Predicho (GPa)')
    plt.title('Pred vs Real')

    plt.grid(True)
    if epoch is not None:
        plt.savefig(f'pred_vs_real_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('pred_vs_real_final.png', dpi=300, bbox_inches='tight')
    plt.close()


"""
def generate_geometry_from_target(eq_predictor, decoder, target_Eeq, latent_dim=32, num_steps=500, lr=1e-2, device='cpu'):

    z = torch.randn((1, latent_dim), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([z], lr=lr)

    target = torch.tensor([target_Eeq], dtype=torch.float32, device=device)

    for step in range(num_steps):
        optimizer.zero_grad()
        pred = eq_predictor(z).squeeze()
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.6f} | Pred: {pred.item():.4f} | Target: {target_Eeq}")

    # Obtener geometría
    dummy_batch = torch.zeros(1, dtype=torch.long, device=device)  # una muestra
    recon_nodes = decoder(z, dummy_batch, None)  # None si no usas ref_coords
    return z.detach(), recon_nodes.detach()
"""

def save_top_k_errors_to_csv(E_eq_pred, E_eq_real, batch_data, epoch, k=5, filename_prefix='top_errors'):
    abs_errors = torch.abs(E_eq_pred - E_eq_real)
    topk_values, topk_indices = torch.topk(abs_errors, k)

    rows = []
    for i in range(k):
        idx = topk_indices[i].item()
        rows.append({
            'rank': i + 1,
            'pred': E_eq_pred[idx].item(),
            'real': E_eq_real[idx].item(),
            'abs_error': abs(E_eq_pred[idx] - E_eq_real[idx]).item(),
            'relative_error_%': 100 * abs(E_eq_pred[idx] - E_eq_real[idx]).item() / (E_eq_real[idx].item() + 1e-6),
            'density': batch_data.y[idx, 1].item() if batch_data.y.shape[1] > 1 else None
        })

    df = pd.DataFrame(rows)
    csv_path = f'{filename_prefix}_epoch_{epoch}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Guardado: {csv_path}")


def save_metrics_history(metrics_history, filename="metrics_per_epoch.csv"):
    df = pd.DataFrame(metrics_history)
    df.to_csv(filename, index=False)
    print(f" Guardado CSV de métricas: {filename}")



# --- 3. Visualización Pred vs Real acumulado ---
def scatter_pred_vs_real(pred, real, epoch=None):
    pred_np = pred.detach().cpu().numpy().flatten()
    real_np = real.detach().cpu().numpy().flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(real_np, pred_np, alpha=0.6, c='blue')
    plt.plot([real_np.min(), real_np.max()], [real_np.min(), real_np.max()], 'r--')
    plt.xlabel('E_eq Real (GPa)')
    plt.ylabel('E_eq Predicho (GPa)')
    plt.title('Pred vs Real')
    plt.grid(True)

    if epoch is not None:
        plt.savefig(f'pred_vs_real_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('pred_vs_real_final.png', dpi=300, bbox_inches='tight')
    plt.close()
