import torch
import torch.nn as nn
import numpy as np
import os
import json
from scipy.spatial import Delaunay
from full_model_fss_decoder import FullNodeDecoder
from full_model_fss_encoder_based import GCAE

# --- Modelos auxiliares ---
class ZGenerator(nn.Module):
    def __init__(self, noise_dim=64, latent_dim=32, n_nodes=56):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_nodes * latent_dim)
        )
        self.n_nodes = n_nodes
        self.latent_dim = latent_dim

    def forward(self, noise):
        return self.net(noise).view(-1, self.n_nodes, self.latent_dim)

class ZModulator(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, z, eq):
        B, N, D = z.shape
        eq_exp = eq.view(B, 1, 1).expand(B, N, 1)
        return self.net(torch.cat([z, eq_exp], dim=-1))

# --- Configuración ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "generated_lattices_json"
os.makedirs(SAVE_DIR, exist_ok=True)

LATENT_DIM = 32
N_NODES = 56
NOISE_DIM = 64

# --- Carga modelos ---
zgen = ZGenerator(NOISE_DIM, LATENT_DIM, N_NODES).to(DEVICE)
zmod = ZModulator(LATENT_DIM).to(DEVICE)
decoder = FullNodeDecoder(LATENT_DIM).to(DEVICE)
gcae = GCAE().to(DEVICE)

zgen.load_state_dict(torch.load("zgen_training_outputs/zgen_latest.pth", map_location=DEVICE))
zmod.load_state_dict(torch.load("zmodulator_training_outputs/zmodulator_latest.pth", map_location=DEVICE))
decoder.load_state_dict(torch.load("best_decoder_dual.pth", map_location=DEVICE))
gcae.load_state_dict(torch.load("best_encoder_predictor_nopool.pth", map_location=DEVICE))

zgen.eval()
zmod.eval()
decoder.eval()
gcae.eval()
predictor = gcae.predictor

# --- Geometría del cubo ---
CUBE_CORNERS = np.array([
    [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0],
    [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]
])
CUBE_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3),
              (4, 5), (4, 6), (5, 7), (6, 7),
              (0, 4), (1, 5), (2, 6), (3, 7)]

# --- Función principal ---
def generate_and_save(eq_val, idx):
    with torch.no_grad():
        eq_tensor = torch.tensor([[eq_val]], dtype=torch.float32, device=DEVICE)
        noise = torch.randn(1, NOISE_DIM).to(DEVICE)

        # Paso 1: Generar z y modificarlo
        z = zgen(noise)
        z_mod = zmod(z, eq_tensor)

        # Paso 2: Decodificar a nodos 3D
        coords = decoder(z_mod.view(-1, LATENT_DIM)).view(1, N_NODES, 3)
        cmin = coords.amin(dim=1, keepdim=True)
        cmax = coords.amax(dim=1, keepdim=True)
        coords = (coords - cmin) / (cmax - cmin + 1e-6)
        coords = coords * 0.9 + 0.05  # [0.05, 0.95]^3

        coords_np = coords[0].cpu().numpy()
        all_nodes = np.vstack([CUBE_CORNERS, coords_np])  # [8 + N, 3]

        # Paso 3: Delaunay + edges del cubo
        tri = Delaunay(all_nodes)
        edge_set = set()
        for simplex in tri.simplices:
            for i in range(4):
                for j in range(i+1, 4):
                    edge = tuple(sorted((simplex[i], simplex[j])))
                    edge_set.add(edge)

        edge_set.update(CUBE_EDGES)
        edges = [ [int(i), int(j)] for i, j in edge_set ]

        # Paso 4: Predecir módulo
        batch = torch.zeros(N_NODES, dtype=torch.long, device=DEVICE)
        eq_pred = predictor(z_mod.view(-1, LATENT_DIM), batch).cpu().item()

        # Paso 5: Guardar JSON
        lattice = {
            "nodes": all_nodes.tolist(),
            "edges": edges,
            "equivalent_youngs_modulus": eq_pred,
            "target_youngs_modulus": float(eq_val)
        }
        outpath = os.path.join(SAVE_DIR, f"lattice_{idx:03d}.json")
        with open(outpath, 'w') as f:
            json.dump(lattice, f, indent=2)
        print(f"[{idx}] Saved: {outpath} | Target: {eq_val:.3f} | Pred: {eq_pred:.3f}")

# --- CLI ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eq_values", nargs="+", type=float, required=True)
    args = parser.parse_args()

    for i, eq in enumerate(args.eq_values):
        generate_and_save(eq, i)
