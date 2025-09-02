import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import DataLoader

from utils_train_full import *
from processing_data import LatticeDataset, custom_collate_fn
from full_model_fss_encoder_based import GCAE  # Nuevo encoder entrenado sin pooling

# ---------- Config ----------
SAVE_DIR = 'reconstructions_dual'
os.makedirs(SAVE_DIR, exist_ok=True)
METRIC_CSV = os.path.join(SAVE_DIR, 'decoder_metrics.csv')

# ---------- Decoder ----------
class FullNodeDecoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=128, node_dim=3):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, z):
        return self.decoder(z)

# ---------- Visualización ----------
def plot_reconstruction(x_real, x_recon, epoch, sample_idx=0):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title(f"Recon vs Original (Epoch {epoch})")
    ax.scatter(*x_real.T, color='blue', label='Original', alpha=0.5)
    ax.scatter(*x_recon.T, color='red', label='Reconstruido', alpha=0.5)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=30)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"overlay_epoch{epoch:03d}_sample{sample_idx}_V2.png"))
    plt.close()

# ---------- Evaluación personalizada ----------
def compute_metrics(pred_coords, true_coords):
    abs_error = torch.abs(pred_coords - true_coords)
    mae = abs_error.mean().item()

    threshold = 0.02  # 2% de una arista de longitud 1
    distances = torch.norm(pred_coords - true_coords, dim=1)
    accurate = (distances < threshold).sum().item()
    accuracy_rate = 100.0 * accurate / true_coords.size(0)

    return mae, accuracy_rate

# ---------- Entrenamiento ----------
def train_decoder(latent_dim=32, batch_size=32, num_epochs=301):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = LatticeDataset(root='data/lattices')
    _, _, test_loader = create_data_loaders(dataset, batch_size=batch_size)

    encoder = GCAE().to(device)
    encoder.load_state_dict(torch.load('best_encoder_predictor_nopool.pth'))
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    decoder = FullNodeDecoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')

    # Preparar archivo CSV
    with open(METRIC_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'mse_loss', 'mae', 'accuracy_percent'])

    for epoch in range(1, num_epochs+1):
        decoder.train()
        total_loss = 0
        total_mae = 0
        total_acc = 0
        total_nodes = 0

        for batch in test_loader:
            batch = batch.to(device)
            x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
            h = encoder.encoder(x, edge_index, edge_attr)
            pred_coords = decoder(h)
            true_coords = batch.ref_coords[:, :3]

            loss = loss_fn(pred_coords, true_coords)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            mae, acc = compute_metrics(pred_coords, true_coords)
            total_mae += mae * true_coords.size(0)
            total_acc += acc * true_coords.size(0)
            total_nodes += true_coords.size(0)

        avg_loss = total_loss / len(test_loader)
        avg_mae = total_mae / total_nodes
        avg_acc = total_acc / total_nodes

        print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.6f} | MAE: {avg_mae:.6f} | Accuracy (<2%): {avg_acc:.2f}%")

        # Guardar mejor modelo
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(decoder.state_dict(), 'best_decoder_dual.pth')

        # Guardar métricas en CSV
        with open(METRIC_CSV, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, avg_mae, avg_acc])

        # Visualización
        if epoch % 10 == 0:
            decoder.eval()
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    h = encoder.encoder(batch.x, batch.edge_index, batch.edge_attr)
                    pred = decoder(h)

                    batch_size = batch.num_graphs
                    node_ptr = batch.ptr

                    for i in range(min(3, batch_size)):
                        start, end = node_ptr[i].item(), node_ptr[i+1].item()
                        true_coords = batch.ref_coords[start:end, :3].cpu().numpy()
                        pred_coords = pred[start:end].cpu().numpy()
                        plot_reconstruction(true_coords, pred_coords, epoch, sample_idx=i)
                    break

if __name__ == '__main__':
    train_decoder()
