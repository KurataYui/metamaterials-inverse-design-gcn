import torch
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils_train_full import create_data_loaders
from processing_data import LatticeDataset
from full_model_fss_encoder_based import GCAE

def get_latent_space(model, dataloader, device):
    zs, ys = [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, z = model(batch)
            batch_size = batch.y.size(0)
            pooled = torch.zeros(batch_size, z.size(1), device=device)
            pooled = pooled.index_add(0, batch.batch, z) / torch.bincount(batch.batch, minlength=pooled.size(0)).view(-1, 1)
            zs.append(pooled.cpu())
            ys.append(batch.y[:, 0].cpu())
    return torch.cat(zs), torch.cat(ys)

def plot_latent(zs, ys, title="Latent Space", save_path="checkpoints"):
    zs = zs.numpy()
    ys = ys.numpy()
    reducer = PCA(n_components=2)
    zs = reducer.fit_transform(zs)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(zs[:, 0], zs[:, 1], c=ys, cmap='viridis', s=10)
    plt.colorbar(scatter, label="E_eq")
    plt.title(title)
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LatticeDataset(root='data/lattices')
    _, _, test_loader = create_data_loaders(dataset, batch_size=32)

    output_dir = "latent_evolution"
    os.makedirs(output_dir, exist_ok=True)

    model = GCAE().to(device)

    for epoch in range(30, 202, 30):  # 30, 60, ..., 180, 201
        model_path = f"checkpoints/model_epoch_{epoch:03d}.pth"
        save_path = os.path.join(output_dir, f"latent_epoch_{epoch:03d}.png")

        if not os.path.exists(model_path):
            print(f"Modelo no encontrado: {model_path}")
            continue

        model.load_state_dict(torch.load(model_path, map_location=device))
        zs, ys = get_latent_space(model, test_loader, device)
        plot_latent(zs, ys, title=f"Latent Space - Epoch {epoch}", save_path=save_path)
        print(f"Guardado: {save_path}")
