import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def visualize_lattice(filepath, save_dir="visualizations"):
    os.makedirs(save_dir, exist_ok=True)

    with open(filepath, 'r') as f:
        data = json.load(f)

    nodes = np.array(data["nodes"])
    edges = np.array(data["edges"])
    eq_target = data.get("target_youngs_modulus", -1)
    eq_pred = data.get("equivalent_youngs_modulus", -1)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Nodos
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='red', alpha=0.6)

    # Aristas
    lines = [[nodes[i], nodes[j]] for i, j in edges]
    ax.add_collection3d(Line3DCollection(lines, colors='blue', alpha=0.3, linewidths=1))

    ax.set_title(f"E_eq target: {eq_target:.3f} | pred: {eq_pred:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=20, azim=30)
    plt.tight_layout()

    fname = os.path.splitext(os.path.basename(filepath))[0] + ".png"
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()
    print(f"[OK] Saved {fname} | target: {eq_target:.3f}, pred: {eq_pred:.3f}")

def visualize_all(json_dir="generated_lattices_json"):
    files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    for f in files:
        path = os.path.join(json_dir, f)
        visualize_lattice(path)

if __name__ == "__main__":
    visualize_all()
