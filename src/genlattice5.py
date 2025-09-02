import argparse 
import pickle
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import json
import os
import os.path as osp
from pathlib import Path
from typing import Dict, List, Tuple
import gc
from multiprocessing import Pool, freeze_support, cpu_count
from tqdm import tqdm

def init_pool_processes():
    np.random.seed()

class LatticeGenerator:
    def __init__(self,min_E_eq, max_E_eq, min_nodes=50,max_nodes=50, domain_size=1.0, delta=0.05,job_id:int=0):
        self.min_E_eq = min_E_eq
        self.max_E_eq = max_E_eq
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.domain_size = domain_size
        self.delta = delta
        self.youngs_modulus=193e9
        self.diameter=0.01
        self.all_nodes = None
        self.all_edges = None
        self.edge_props = []
        self.num_corners = 0
        self.E_eq = 0
        self.rho_rel = 0
        self.lattices = []
        self.job_id=job_id

    def generate_lattice(self):
        corners = np.array([
            [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0],
            [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]
        ]) * self.domain_size
    
        internal_nodes = []
        num_internal_nodes = np.random.randint(self.min_nodes,self.max_nodes + 1) if (self.max_nodes - self.min_nodes>0) else self.max_nodes
    
        while len(internal_nodes) < num_internal_nodes:
            new_node = (np.random.rand(3) * (1 - 2*self.delta) + self.delta) * self.domain_size
            if len(internal_nodes)==0:
                internal_nodes.append(new_node)
            # Verificar contra todos los nodos existentes (incluyendo esquinas)
            all_nodes = np.vstack([corners, np.array(internal_nodes)])
        
            # Usar distancia euclidiana con tolerancia (1e-6)
            distances = np.linalg.norm(all_nodes - new_node, axis=1)
            if np.all(distances > 1e-6):
                internal_nodes.append(new_node)
    
        self.all_nodes = np.vstack([corners, internal_nodes])
        # Aplicar triangulación de Delaunay 3D
        tri = Delaunay(self.all_nodes)
        
        # Extraer las aristas (evitando duplicados)
        edges = set()
        for simplex in tri.simplices:
            for i in range(4):
                for j in range(i+1, 4):
                    edge = tuple(sorted((simplex[i], simplex[j])))
                    edges.add(edge)

        self.all_edges = np.array([list(edge) for edge in edges])
        self.num_corners = len(corners)

        return self.all_nodes, self.all_edges, self.num_corners

    def compute_stiffness_matrix(self, E=193e9, d=0.01):
        """Calcula la matriz de rigidez global"""
        A = np.pi*(d/2)**2  # Área transversal
        num_nodes = len(self.all_nodes)
        K = np.zeros((3*num_nodes, 3*num_nodes))
        
        for edge in self.all_edges:
            i, j = edge
            xi, yi, zi = self.all_nodes[i]
            xj, yj, zj = self.all_nodes[j]
            
            # Longitud y cosenos directores
            L = np.sqrt((xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2)
            cx = (xj-xi)/L
            cy = (yj-yi)/L
            cz = (zj-zi)/L
            # funciona hasta aquí
            # Matriz de transformación
            T = np.array([
                [cx, cy, cz, 0, 0, 0],
                [0, 0, 0, cx, cy, cz]
            ])
            # Matriz de rigidez local y global
            k_local = (E*A/L) * np.array([[1, -1], [-1, 1]])
            k_global = T.T @ k_local @ T
            # Ensamblar en matriz global
            indices = np.array([3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2])
            for ii, idx_i in enumerate(indices):
                for jj, idx_j in enumerate(indices):
                    K[idx_i, idx_j] += k_global[ii, jj] 
        return K

    def compute_edge_properties(self):
        """Calcula las propiedades de cada arista (longitud, radio, área, módulo de Young)"""
        self.edge_props = []
        for edge in self.all_edges:
            i, j = edge
            length = np.linalg.norm(self.all_edges[j] - self.all_edges[i])
            props = {
                "length": length,
                "radius": self.diameter/2,
                "area": np.pi*(self.diameter/2)**2,
                "youngs_modulus": self.youngs_modulus
            }
            self.edge_props.append(props)
        return self.edge_props

    def compute_equivalent_youngs_modulus(self, u_star=0.01):
        """Calcula el módulo de Young equivalente según la metodología del paper"""
        K = self.compute_stiffness_matrix()
        num_nodes = len(self.all_nodes)
        # Identificar nodos según condiciones de contorno (Fig. 1b del paper)
        # Nodos base (z=0) - completamente fijos        
        # Nodos superiores (z=1) con desplazamiento impuesto en z
        top_nodes = [i for i, node in enumerate(self.all_nodes) if abs(node[2] - self.domain_size) < 1e-6]
        # Grados de libertad restringidos
        fixed_dofs = []
        # Nodos base completamente fijos
        fixed_dofs = [0, 1, 2, 3, 5, 7, 8, 11, 12, 13, 15, 19]
        # DOFs con desplazamiento impuesto (z superior)
        loaded_dofs = [3*node+2 for node in top_nodes]
        #hasta aquí tira
        # Todos los DOFs restringidos (fijos + cargados)
        restricted_dofs = np.unique(np.concatenate([fixed_dofs, loaded_dofs]))
        restricted_dofs=restricted_dofs.astype(int)
        free_dofs = np.setdiff1d(np.arange(3*num_nodes), restricted_dofs)
        # Submatrices

        K_ff = K[np.ix_(free_dofs, free_dofs)]
        K_fr = K[np.ix_(free_dofs, restricted_dofs)]
        K_rf = K[np.ix_(restricted_dofs, free_dofs)]
        K_rr = K[np.ix_(restricted_dofs, restricted_dofs)]
        
        # Vector de desplazamientos impuestos
        u_r = np.zeros(len(restricted_dofs))
        # Aplicar desplazamiento u_star en los DOFs cargados
        u_r[-len(loaded_dofs):] = -u_star  # Negativo para tracción
        
        # Resolver sistema
        u_f = np.linalg.solve(K_ff, -K_fr @ u_r)
        
        # Ensamblar solución completa
        u = np.zeros(3*num_nodes)
        u[free_dofs] = u_f
        u[restricted_dofs] = u_r
        
        # Calcular fuerzas de reacción
        f_r = K @ u
        
        # Calcular trabajo de fuerzas externas (solo en DOFs cargados)
        W_ext = np.dot(f_r[loaded_dofs], u[loaded_dofs])
        
        # Calcular módulo equivalente (según ecuación 11 del paper)
        L = self.domain_size
        A_cross = self.domain_size**2  # Área transversal
        self.E_eq = (W_ext / (u_star * A_cross)) * (L / u_star)
        
        # Calcular densidad relativa
        V_total = self.domain_size**3
        V_material = sum(np.linalg.norm(self.all_nodes[e[1]] - self.all_nodes[e[0]]) * np.pi*(0.01/2)**2 for e in self.all_edges)

        self.rho_rel = V_material / V_total
        
        return self.E_eq, self.rho_rel

    def add_lattice_to_dict(self):
        E_eq_test = (self.min_E_eq <= self.E_eq < self.max_E_eq)
        if E_eq_test:
            lattice_data = {
                "nodes": self.all_nodes.tolist() if isinstance(self.all_nodes, np.ndarray) else self.all_nodes,
                "edges": self.all_edges.tolist() if isinstance(self.all_edges, np.ndarray) else self.all_edges,
                "edge_properties": self.edge_props,
                "equivalent_youngs_modulus": self.E_eq,
                "relative_density": self.rho_rel
            }
            self.lattices.append(lattice_data)

        return E_eq_test

    def plot_lattice(self):
        """Visualiza la estructura de lattice en 3D con restricciones en vértices"""
        fixed_dofs = [0, 1, 2, 3, 5, 7, 8, 11, 12, 13, 15, 19]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Dibujar nodos normales
        ax.scatter(self.all_nodes[:, 0], self.all_nodes[:, 1], self.all_nodes[:, 2], c='b', s=20, label='Nodos libres')

    # Dibujar aristas
        edge_lines = [[self.all_nodes[e[0]], self.all_nodes[e[1]]] for e in self.all_edges]
        edge_collection = Line3DCollection(edge_lines, linewidths=1, colors='k')
        ax.add_collection3d(edge_collection)

    # Coordenadas de los vértices del cubo (escaladas al dominio)
        corners_coord = np.array([
            [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0],
            [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]
        ]) * self.domain_size

    # Dibujar restricciones en cada vértice
        for node_idx in range(8):  # 8 vértices del cubo
            x, y, z = corners_coord[node_idx]
            gdl_x = node_idx * 3
            gdl_y = node_idx * 3 + 1
            gdl_z = node_idx * 3 + 2
        
            marker_size = 150  # Tamaño aumentado para mejor visibilidad
        
            # Dibujar marcadores solo para los GDLs restringidos
            if gdl_x in fixed_dofs:
                ax.scatter(x, y, z, marker='>', s=marker_size, c='red', alpha=0.8, 
                      label='Restricción X' if node_idx == 0 else "")
            if gdl_y in fixed_dofs:
                ax.scatter(x, y, z, marker='^', s=marker_size, c='green', alpha=0.8, 
                      label='Restricción Y' if node_idx == 0 else "")
            if gdl_z in fixed_dofs:
                ax.scatter(x, y, z, marker='1', s=marker_size, c='blue', alpha=0.8, 
                      label='Restricción Z' if node_idx == 0 else "")

        # Configuración de ejes
        ax.set_xlim([0, self.domain_size])
        ax.set_ylim([0, self.domain_size])
        ax.set_zlim([0, self.domain_size])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Estructura de Lattice ({len(self.all_nodes)} nodos, {len(self.all_edges)} aristas)')

    # Mostrar leyenda sin duplicados
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique))

        plt.tight_layout()
        plt.show()
    # Configuración de ejes
        ax.set_xlim([0, self.domain_size])
        ax.set_ylim([0, self.domain_size])
        ax.set_zlim([0, self.domain_size])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Estructura de Lattice ({len(self.all_nodes)} nodos, {len(self.all_edges)} aristas)')

    # Mostrar leyenda solo una vez por cada tipo
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Eliminar duplicados
        ax.legend(by_label.values(), by_label.keys())
    
        """plt.tight_layout()
        plt.show()
"""

class DatasetGenerator:
    def __init__(self,min_E_eq, max_E_eq,output_dir="data/lattices",num_samples=32, min_nodes=50,max_nodes=50, domain_size=1.0, delta=0.05):
        self.min_E_eq = min_E_eq
        self.max_E_eq = max_E_eq
        self.output_dir=output_dir
        self.output_filename = os.path.join(self.output_dir, "all_lattices.pkl")
        self.num_samples=num_samples
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.domain_size = domain_size
        self.delta = delta
        self.samples = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def __set_num_sample_in_pickle(self,f):
        pickle.dump(self.num_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Setting {self.num_samples} of samples in lattice file {self.output_filename}")

    def __update_pickle(self,f):
        for lattices in tqdm (self.all_lattices,desc="Dumping lattices to file.",dynamic_ncols=True,leave=False):
            pickle.dump(lattices, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            del lattices

        if self.all_lattices:
            del self.all_lattices
        self.all_lattices = []
        gc.collect()

    def __build_workers(self,jobs:int):
        workers = []
        for j in range(jobs):
            workers.append(LatticeGenerator(
                job_id=j+1,
                min_E_eq=self.min_E_eq,
                 max_E_eq=self.max_E_eq,
                min_nodes=self.min_nodes,
                max_nodes=self.max_nodes,
                domain_size=self.domain_size,
                delta=self.delta))
        return workers

    def _run_lattice_batch(self,lattice_gen):
        for i in tqdm (range(self.samples), desc=f"Job {lattice_gen.job_id} Generating lattices",position=lattice_gen.job_id,total=self.samples,dynamic_ncols=True,leave=False):
            lattice_ok=False
            while (not lattice_ok):
                lattice_gen.generate_lattice()
                lattice_gen.compute_equivalent_youngs_modulus()
                lattice_gen.compute_edge_properties()
                lattice_ok = lattice_gen.add_lattice_to_dict()

        return lattice_gen.lattices

    def generate_dataset(self,update_batch:int,cpu_fraction:int):
        """Genera dataset con validación de estabilidad"""
        jobs=int(cpu_count()*cpu_fraction)
        update_batch = self.num_samples if self.num_samples < update_batch else update_batch
        num_batches = int(self.num_samples/update_batch) + (self.num_samples % update_batch > 0)
        self.samples = int(update_batch/jobs) + (update_batch % jobs > 0)
        workers = tuple(self.__build_workers(jobs))
        successful_samples = 0
        freeze_support()
        with open(self.output_filename, 'wb') as f:
            self.__set_num_sample_in_pickle(f)
            for _ in tqdm (range(num_batches), desc=f"Total batches executions",position=0,total=num_batches,dynamic_ncols=True):
                with Pool(jobs,initializer=init_pool_processes) as p:
                    self.all_lattices = p.map(self._run_lattice_batch,workers)
                self.__update_pickle(f)
                successful_samples += update_batch

            if self.all_lattices:
                self.__update_pickle()

        print(f"All {self.num_samples} lattices saved to {self.output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Lattice Generatoe',
                        epilog='Text at the bottom of help')
    parser.add_argument("num_samples", help="number of examples to generate",
                    type=int, nargs='?',default=50000)
    parser.add_argument("min_e_eq", help="Lower limit for equivalent_youngs_modulus",
                    type=float, nargs='?',default=80000000.00)
    parser.add_argument("max_e_eq", help="Upper limit for equivalent_youngs_modulus",
                    type=float, nargs='?',default=120000000.00)
    parser.add_argument("cpu_fraction", help="Fraction of cpu power used",
                    type=float, nargs='?',default=0.4)
    parser.add_argument("min_nodes", help="minimum number of nodes per lattice",
                    type=int, nargs='?',default=10)
    parser.add_argument("max_nodes", help="maximun number of nodes per lattice",
                    type=int, nargs='?',default=50)
    parser.add_argument("update_batch", help="Number lattice saved to file on each write operation",
                    type=int, nargs='?',default=50000)
    parser.add_argument("--output_dir", help="output directory for data",
                    type=str,default="data/lattices",required=False)
    parser.add_argument("--use_pickle", help="use pkl format",
                    type=bool,default=True,required=False)
    parser.add_argument("--plot_lattice", help="Plot a lattice a 50 cells,for running inside Jupyter notebook",
                    type=bool,default=False,required=False)
    args = parser.parse_args()

    print("Parameters:")
    print(f"min_e_eq:{args.min_e_eq}")
    print(f"max_e_eq:{args.max_e_eq}")
    print(f"plot_lattice:{args.plot_lattice}")
    print(f"num_samples:{args.num_samples}")
    print(f"cpu_fraction:{args.cpu_fraction}")
    print(f"min_nodes:{args.min_nodes}")
    print(f"max_nodes:{args.max_nodes}")
    print(f"output_dir:{args.output_dir}")
    print(f"update_batch:{args.update_batch}")
    print("")

    assert (args.cpu_fraction <= 0.7 and args.cpu_fraction > 0),"cpu_fraction must be between 0 and 0.7"

    if args.plot_lattice:
        generator=LatticeGenerator(min_E_eq=args.min_e_eq, max_E_eq=args.max_e_eq)
        generator.generate_lattice()
        generator.compute_equivalent_youngs_modulus()
        print("Ignoring generation parameters.")
        print("No output file.")
        generator.plot_lattice()
    else:
        generator =  DatasetGenerator(
            min_E_eq=args.min_e_eq,
            max_E_eq=args.max_e_eq,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            output_dir=args.output_dir,
            num_samples=args.num_samples
        )
        generator.generate_dataset(args.update_batch,args.cpu_fraction)

        script_dir = Path(__file__).parent
        data_dir = script_dir / "data" / "lattices"

        # Asegurar estructura de directorios
        (data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (data_dir / "processed").mkdir(parents=True, exist_ok=True)

        # Mover/copiar el archivo .pkl a raw/ si no está allí
        pkl_src = data_dir / "all_lattices.pkl"
        pkl_dst = data_dir / "raw" / "all_lattices.pkl"
        if not pkl_dst.exists() and pkl_src.exists():
            import shutil
            shutil.copy(pkl_src, pkl_dst)
