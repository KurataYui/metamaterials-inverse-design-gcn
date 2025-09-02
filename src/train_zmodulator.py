import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from full_model_fss_decoder import FullNodeDecoder
from full_model_fss_encoder_based import GCAE

# Configuración
LATENT_DIM = 32
N_NODES = 56
NOISE_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "zmodulator_training_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# Cargar decoder (congelado)
decoder = FullNodeDecoder(latent_dim=LATENT_DIM).to(DEVICE)
decoder.load_state_dict(torch.load("best_decoder_dual.pth", map_location=DEVICE))
decoder.eval()
for p in decoder.parameters():
    p.requires_grad = False

# Cargar predictor (congelado)
gcae = GCAE().to(DEVICE)
gcae.load_state_dict(torch.load("best_encoder_predictor_nopool.pth", map_location=DEVICE))
gcae.eval()
predictor = gcae.predictor
for p in predictor.parameters():
    p.requires_grad = False

# Cargar generador z preentrenado
class ZGenerator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM, latent_dim=LATENT_DIM, n_nodes=N_NODES, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_nodes * latent_dim)
        )

    def forward(self, noise):
        z = self.net(noise)
        return z.view(-1, N_NODES, LATENT_DIM)

zgen = ZGenerator().to(DEVICE)
zgen.load_state_dict(torch.load("zgen_training_outputs/zgen_latest.pth", map_location=DEVICE))
zgen.eval()

# ZModulator: modifica z_geom según E_eq
class ZModulator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, z_in, eq_target):
        B, N, D = z_in.shape
        eq_expanded = eq_target.view(B, 1, 1).expand(B, N, 1)
        input_cat = torch.cat([z_in, eq_expanded], dim=-1)
        return self.net(input_cat)

# Instanciar y optimizar
modulator = ZModulator().to(DEVICE)
optimizer = torch.optim.Adam(modulator.parameters(), lr=1e-3)

# Entrenamiento
def train_zmodulator(steps=1000, batch_size=64):
    for step in range(1, steps + 1):
        z_noise = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
        eq_target = torch.rand(batch_size, 1).to(DEVICE)  # valores deseados de E_eq ∈ [0, 1]

        with torch.no_grad():
            z_geom = zgen(z_noise)  # [B, N, D]

        z_mod = modulator(z_geom, eq_target)  # [B, N, D]
        coords = decoder(z_mod.view(-1, LATENT_DIM)).view(batch_size, N_NODES, 3)

        # Normalizar coordenadas a [0.05, 0.95] por eje
        coords_min = coords.amin(dim=1, keepdim=True)  # [B, 1, 3]
        coords_max = coords.amax(dim=1, keepdim=True)  # [B, 1, 3]
        range_ = coords_max - coords_min + 1e-6
        coords = (coords - coords_min) / range_          # [0, 1]
        coords = coords * 0.9 + 0.05                      # [0.05, 0.95]

        batch_vec = torch.arange(batch_size, device=DEVICE).repeat_interleave(N_NODES)
        pred_eq = predictor(z_mod.view(-1, LATENT_DIM), batch_vec)

        loss = F.mse_loss(pred_eq, eq_target.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"[Step {step}] Eq Loss: {loss.item():.6f} | Target: {eq_target[0].item():.3f} | Pred: {pred_eq[0].item():.3f}")
            coords_np = coords[0].detach().cpu().numpy()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2], c='blue')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.set_title(f"Step {step} | Eeq Target {eq_target[0].item():.3f}")
            plt.savefig(os.path.join(SAVE_DIR, f"generated_{step:04d}.png"))
            plt.close()

        if step % 200 == 0:
            torch.save(modulator.state_dict(), os.path.join(SAVE_DIR, "zmodulator_latest.pth"))

if __name__ == "__main__":
    train_zmodulator()
