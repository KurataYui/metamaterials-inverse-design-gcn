import os
import json
import numpy as np
import argparse
from genlattice5 import LatticeGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def validate_json_folder(folder):
    results = []

    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    files.sort()

    for filename in files:
        path = os.path.join(folder, filename)

        with open(path, "r") as f:
            data = json.load(f)

        nodes = np.array(data["nodes"], dtype=np.float64)
        edges = np.array(data["edges"], dtype=np.int32)

        # Creamos una instancia dummy de LatticeGenerator
        latgen = LatticeGenerator(min_E_eq=0, max_E_eq=1)
        latgen.all_nodes = nodes
        latgen.all_edges = edges
        latgen.domain_size = 1.0

        try:
            E_eq, rho_rel = latgen.compute_equivalent_youngs_modulus()

            target_E = data.get("target_youngs_modulus", None)
            if target_E is not None:
                target_E *= 163.73e6  # Escalado

            rel_error = abs(E_eq - target_E) / target_E
            print(f"[{filename}]  Target: {target_E:.2e} | Computed: {E_eq:.2e} | Rel. Error: {rel_error:.2e}")

            results.append({
                "file": filename,
                "target": target_E,
                "computed": E_eq,
                "error": abs(E_eq - target_E),
                "rel_error": rel_error
            })

        except Exception as e:
            print(f"[{filename}] Error computing FEM: {e}")

    # Mostrar los 5 mejores
    if results:
        print("\nTop 5 mejores resultados (menor error relativo):")
        results.sort(key=lambda x: x["rel_error"])
        for r in results[:5]:
            print(f"{r['file']} -> Rel. Error: {r['rel_error']:.2e} | Target: {r['target']:.2e} | Computed: {r['computed']:.2e}")

        # Guardar CSV
        csv_path = os.path.join(folder, "results.csv")
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Resultados guardados en {csv_path}")

    return results


def plot_stiffness_matrix(K_global, title='Ocupación de matriz de rigidez global', threshold=1e-6):
    """
    Visualiza la ocupación (valores relevantes) de la matriz de rigidez global como un heatmap binario o continuo.
    
    Parámetros:
    - K_global: np.ndarray o scipy.sparse matriz cuadrada.
    - title: título del gráfico.
    - threshold: valores menores a este umbral se consideran cero.
    """
    if not isinstance(K_global, np.ndarray):
        K = K_global.toarray()
    else:
        K = K_global.copy()

    # Para visualización: valores absolutos y normalización opcional
    K_abs = np.abs(K)
    K_mask = (K_abs > threshold).astype(float)  # binarización opcional

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(K_mask, cmap='viridis', cbar=True,
                xticklabels=False, yticklabels=False,
                square=True, linewidths=0)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Path to folder with lattice_*.json files")
    args = parser.parse_args()

    validate_json_folder(args.folder)

#python generate_lattice_from_z.py --eq_values 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95