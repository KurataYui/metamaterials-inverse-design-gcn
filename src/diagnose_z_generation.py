import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from full_model_fss_decoder import FullNodeDecoder

# Config
LATENT_DIM = 32
N_NODES = 56
NOISE_DIM = 64
SAVE_DIR = "zgen_training_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ZGenerator(nn.Module):
    def __init__(self, noise_dim=64, latent_dim=32, n_nodes=56, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_nodes * latent_dim)
        )
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes

    def forward(self, noise):
        z = self.net(noise)  # [B, N * D]
        return z.view(-1, self.n_nodes, self.latent_dim)

# Decoder (congelado)
decoder = FullNodeDecoder(latent_dim=LATENT_DIM).to(DEVICE)
decoder.load_state_dict(torch.load("best_decoder_dual.pth", map_location=DEVICE))
decoder.eval()
for p in decoder.parameters():
    p.requires_grad = False

# Instanciar generador
zgen = ZGenerator().to(DEVICE)
optimizer = torch.optim.Adam(zgen.parameters(), lr=1e-3)

# Pérdidas
def loss_fn(coords):
    B = coords.size(0)
    out_bounds = torch.clamp(coords - 1, min=0) + torch.clamp(-coords, min=0)
    loss_bounds = out_bounds.abs().mean()
    collapse = 0.0
    for b in range(B):
        dists = torch.cdist(coords[b], coords[b])
        mask = (dists < 1e-3) & (~torch.eye(N_NODES, device=DEVICE).bool())
        collapse += mask.sum()
    loss_collapse = collapse / B
    return loss_bounds + 0.001 * loss_collapse, loss_bounds.item(), loss_collapse.item()

# Visualización
def plot_coords(coords, step):
    coords = coords.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:,0], coords[:,1], coords[:,2], c='red')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_title(f'Sample at step {step}')
    plt.savefig(os.path.join(SAVE_DIR, f"sample_{step:04d}.png"))
    plt.close()

# Entrenamiento
def train(num_steps=1000, batch_size=64):
    for step in range(1, num_steps+1):
        z_noise = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
        z_gen = zgen(z_noise)  # [B, N, D]
        coords = decoder(z_gen.view(-1, LATENT_DIM)).view(batch_size, N_NODES, 3)

        loss, lb, lc = loss_fn(coords)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"[Step {step}] Loss: {loss.item():.6f} | Bounds: {lb:.4e} | Collapse: {lc:.1f}")
            plot_coords(coords[0], step)

        if step % 200 == 0:
            torch.save(zgen.state_dict(), os.path.join(SAVE_DIR, "zgen_latest.pth"))

if __name__ == "__main__":
    train(num_steps=1000)
