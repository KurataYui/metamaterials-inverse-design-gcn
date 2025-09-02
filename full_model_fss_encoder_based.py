import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
import os
import csv
from processing_data import LatticeDataset
from utils_train_full import create_data_loaders, evaluate, print_now
import matplotlib as plt

class GCELoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        rel_mae = torch.mean(torch.abs(pred - target) / (target + 1e-6))
        vx = pred - pred.mean()
        vy = target - target.mean()
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
        corr_penalty = 1 - corr
        abs_mae = torch.mean(torch.abs(pred - target))
        return rel_mae + self.alpha * corr_penalty + self.beta * abs_mae, abs_mae, rel_mae, corr

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, h):
        return self.encoder(h)

class Encoder(nn.Module):
    def __init__(self, node_dim=3, edge_dim=7, hidden_dim=128, heads=4):
        super().__init__()
        self.gat = GATConv(node_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, concat=True)
        self.encoder = GCNEncoder(hidden_dim, hidden_dim, latent_dim=32)

    def forward(self, x, edge_index, edge_attr):
        h = F.elu(self.gat(x, edge_index, edge_attr))
        z = self.encoder(h)
        return z

class Predictor(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z, batch):
        pooled = torch.zeros(batch.max() + 1, z.size(1), device=z.device)
        pooled = pooled.index_add(0, batch, z) / torch.bincount(batch, minlength=pooled.size(0)).view(-1, 1)
        return self.mlp(pooled).squeeze(-1)

class GCAE(nn.Module):
    def __init__(self, node_dim=3, edge_dim=7, hidden_dim=128, latent_dim=32, heads=4):
        super().__init__()
        self.encoder = Encoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, heads=heads)
        self.predictor = Predictor(latent_dim=latent_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        z = self.encoder(x, edge_index, edge_attr)
        pred = self.predictor(z, batch)
        return pred, z

def plot_metrics(metrics_dict, save_path="metrics_plot.png"):
    epochs = list(range(1, len(metrics_dict['r2']) + 1))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics_dict['r2'], label='R²')
    plt.plot(epochs, metrics_dict['rel_mae'], label='Relative MAE')
    plt.plot(epochs, metrics_dict['abs_mae'], label='Absolute MAE')
    plt.xlabel('Epoch')
    plt.title('Validation Metrics')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics_dict['corr_loss'], label='1 - Correlation')
    plt.xlabel('Epoch')
    plt.title('Correlation Penalty')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(num_epochs=100, batch_size=32):
    print_now("Inicio del entrenamiento con seguimiento de métricas")
    csv_path = "metrics_log.csv"
    
    
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_r2", "val_abs_mae", "val_rel_mae", "val_corr_penalty"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = LatticeDataset(root='data/lattices')
    norm_params = dataset.get_normalization_params()
    train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size=batch_size)

    model = GCAE().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    loss_fn = GCELoss()

    best_r2 = -float('inf')
    metrics = {'r2': [], 'abs_mae': [], 'rel_mae': [], 'corr_loss': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred, _ = model(batch)
            loss, abs_mae, rel_mae, corr = loss_fn(pred, batch.y[:, 0])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            all_preds, all_targets = [], []
            for batch in val_loader:
                batch = batch.to(device)
                pred, _ = model(batch)
                all_preds.append(pred)
                all_targets.append(batch.y[:, 0])
            pred = torch.cat(all_preds)
            target = torch.cat(all_targets)
            abs_mae = torch.mean(torch.abs(pred - target)).item()
            rel_mae = torch.mean(torch.abs(pred - target) / (target + 1e-6)).item()
            vx, vy = pred - pred.mean(), target - target.mean()
            corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
            corr_loss = (1 - corr.item())
            r2 = 1 - torch.sum((pred - target) ** 2) / torch.sum((target - target.mean()) ** 2)

            metrics['r2'].append(r2.item())
            metrics['abs_mae'].append(abs_mae)
            metrics['rel_mae'].append(rel_mae)
            metrics['corr_loss'].append(corr_loss)

        scheduler.step(r2.item())
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss / len(train_loader):.4f} | R²: {r2:.3f} | Abs MAE: {abs_mae:.4f} | Rel MAE: {rel_mae:.4f} | 1-Corr: {corr_loss:.4f}")
        # Guardar métricas por época
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                total_loss / len(train_loader),
                r2.item(),
                abs_mae,
                rel_mae,
                corr_loss
            ])

        #with open('metrics_log.csv', 'a', newline='') as f:
        #    writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 
        #                                        'val_r2', 'val_abs_mae', 'val_rel_mae',
        #                                        'val_corr_penalty', 'lr'])
        #    if epoch == 0:
        #        writer.writeheader()
        #    writer.writerow({
        #        'epoch': epoch,
        #        'train_loss': total_loss,
        #        'val_loss': (abs_mae+rel_mae+corr_loss),
        #        'val_r2': r2.item(),
        #        'val_abs_mae': abs_mae,
        #        'val_rel_mae': rel_mae,
        #        'val_corr_penalty': corr_loss,
        #        'lr': optimizer.param_groups[0]['lr']
        #    })
        if r2 > best_r2:
            best_r2 = r2
             # Crear archivo CSV para guardar métricas
            csv_path = "metrics_log.csv"
            torch.save(model.state_dict(), 'best_encoder_predictor_nopool.pth')

    plot_metrics(metrics, save_path="metrics_summary.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=201)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    train_model(num_epochs=args.epochs, batch_size=args.batch_size)



#class GCELoss(nn.Module):
#    def __init__(self, alpha=0.3, beta=0.5):
#        super().__init__()
#        self.alpha = alpha
#        self.beta = beta
#
#    def forward(self, pred, target):
#        pred = pred.view(-1)
#        target = target.view(-1)
#        rel_mae = torch.mean(torch.abs(pred - target) / (target + 1e-6))
#        vx = pred - pred.mean()
#        vy = target - target.mean()
#        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-6)
#        corr_penalty = 1 - corr
#        abs_mae = torch.mean(torch.abs(pred - target))
#        return rel_mae + self.alpha * corr_penalty + self.beta * abs_mae
#
#
#class GCNEncoder(nn.Module):
#    def __init__(self, in_dim, hidden_dim, latent_dim):
#        super().__init__()
#        self.encoder = nn.Sequential(
#            nn.Linear(in_dim, hidden_dim),
#            nn.PReLU(),
#            nn.Linear(hidden_dim, latent_dim)
#        )
#
#    def forward(self, h):
#        return self.encoder(h)
#
#
#class Encoder(nn.Module):
#    def __init__(self, node_dim=3, edge_dim=7, hidden_dim=128, heads=4):
#        super().__init__()
#        self.gat = GATConv(node_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim, concat=True)
#        self.encoder = GCNEncoder(hidden_dim, hidden_dim, latent_dim=32)
#
#    def forward(self, x, edge_index, edge_attr):
#        h = F.elu(self.gat(x, edge_index, edge_attr))  # [N, hidden_dim]
#        z = self.encoder(h)  # [N, latent_dim]
#        return z
#
#
#class Predictor(nn.Module):
#    def __init__(self, latent_dim=32):
#        super().__init__()
#        self.mlp = nn.Sequential(
#            nn.Linear(latent_dim, 128),
#            nn.SiLU(),
#            nn.Linear(128, 64),
#            nn.SiLU(),
#            nn.Linear(64, 1)
#        )
#
#    def forward(self, z, batch):
#        # Media por grafo (batch-wise)
#        pooled = torch.zeros(batch.max() + 1, z.size(1), device=z.device)
#        pooled = pooled.index_add(0, batch, z) / torch.bincount(batch, minlength=pooled.size(0)).view(-1, 1)
#        return self.mlp(pooled).squeeze(-1)
#
#
#class GCAE(nn.Module):
#    def __init__(self, node_dim=3, edge_dim=7, hidden_dim=128, latent_dim=32, heads=4):
#        super().__init__()
#        self.encoder = Encoder(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, heads=heads)
#        self.predictor = Predictor(latent_dim=latent_dim)
#
#    def forward(self, data):
#        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
#        z = self.encoder(x, edge_index, edge_attr)  # [N, latent_dim]
#        pred = self.predictor(z, batch)  # [B]
#        return pred, z
#
#
#def train_model(num_epochs=100, batch_size=32):
#    print_now("Inicio del entrenamiento Encoder + Predictor (sin pooling)")
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#    dataset = LatticeDataset(root='data/lattices')
#    norm_params = dataset.get_normalization_params()
#    train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size=batch_size)
#
#    model = GCAE().to(device)
#    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
#    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
#    loss_fn = GCELoss()
#
#    best_r2 = -float('inf')
#    for epoch in range(num_epochs):
#        model.train()
#        total_loss = 0
#        for batch in train_loader:
#            batch = batch.to(device)
#            optimizer.zero_grad()
#            pred, _ = model(batch)
#            loss = loss_fn(pred, batch.y[:, 0])
#            loss.backward()
#            optimizer.step()
#            total_loss += loss.item()
#
#        val_metrics = evaluate(model, val_loader, device, epoch, norm_params)
#        scheduler.step(val_metrics['r2'])
#
#        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss / len(train_loader):.4f} | "
#              f"Val R²: {val_metrics['r2']:.3f} | Val MAPE: {val_metrics['mape']:.2f}%")
#
#        if val_metrics['r2'] > best_r2:
#            best_r2 = val_metrics['r2']
#            torch.save(model.state_dict(), 'best_encoder_predictor_nopool.pth')
#
#    print("\nEvaluación en conjunto de test:")
#    model.load_state_dict(torch.load('best_encoder_predictor_nopool.pth'))
#    test_metrics = evaluate(model, test_loader, device, None, norm_params)
#    print(f"  R²: {test_metrics['r2']:.3f} | MAPE: {test_metrics['mape']:.2f}%")
#
#
#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--epochs", type=int, default=201)
#    parser.add_argument("--batch_size", type=int, default=32)
#    args = parser.parse_args()
#
#    train_model(num_epochs=args.epochs, batch_size=args.batch_size)