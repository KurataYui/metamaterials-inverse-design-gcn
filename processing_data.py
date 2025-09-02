import pickle
import torch
from torch_geometric.data import Data, Dataset, Batch
import numpy as np
from typing import List, Optional
import os.path as osp
import os
#from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count
#import torch.multiprocessing as mp
import gc
from datetime import datetime
import pytz
from auxiliar import *
timezone = pytz.timezone("Europe/Madrid")

def print_now(text):
    now = datetime.now(timezone)
    print(f"{text} {now}]")

class LatticeData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ref_coords':
            return 0  # ref_coords no se incrementa como edge_index
        return super().__inc__(key, value, *args, **kwargs)

class LatticeDataset(Dataset):
    def __init__(self, root: str, pkl_file: str = 'all_lattices.pkl', 
                 transform: Optional[callable] = None, pre_transform: Optional[callable] = None):
        self.pkl_file = pkl_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pkl_iter = [] # iterator over the objects in pickle file
        self.lattices_cache = [] #List for pikles on each pivkle objetc
        self.num_files = 0 # initial num of pt files
        self.node_min = None
        self.node_max = None
        self.pkl_path = osp.join(osp.join(root, 'raw'), self.pkl_file)  # Tenía paréntesis mal ubicados
        self._setup_num_files()
        self._calculate_normalization()  # Añade esto
        self.node_scaler = None
        super().__init__(root, transform, pre_transform)
        self.processed_files = self.process()

    def _setup_num_files(self):
        """Lee el primer elemento del archivo para obtener el conteo total"""
        with open(self.pkl_path, 'rb') as f:
            self.num_files = pickle.load(f)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.pkl_file]

    @property
    def processed_file_names(self) -> List[str]:
        return [f'data_{i}.pt' for i in range(self.num_files)]

    def _is_processed(self):
        return len(os.listdir(self.processed_dir)) > 0

    def download(self):
        pass
    
    def get_normalization_params(self):
        """Devuelve los parámetros para desescalar"""
        return {'max': self.e_eq_max}

    def process(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        print_now("Starting process:")

        print(self.num_files)
        num_cpu = int(os.cpu_count()/3)
        global_idx = 0
        processed_files = 0
        with open(self.pkl_path, 'rb') as f:
            pickle.load(f)  # Saltar contador
            while True:
                try:
                    chunk = pickle.load(f)
                    chunk_size = len(chunk)

                    num_workers = num_cpu if chunk_size > num_cpu else chunk_size
                    worker_chunk_size = int(chunk_size // num_workers)
                    chunks = [(i, min(i+worker_chunk_size, chunk_size)) 
                            for i in range(0, chunk_size, worker_chunk_size)]
                    with Pool(num_workers) as pool:
                        resp = pool.starmap(
                            self._process_chunk,
                            [(global_idx + start, chunk[start:end]) for start, end in chunks
                        ])
                    processed_files += sum(resp)
                    global_idx += chunk_size
                    del chunk
                except EOFError:
                    break
        print_now(f"Processed files {processed_files}")
        return processed_files

    def _calculate_normalization(self):
        # Se proceaa en CPU, las pruebas con GeForce RTX 3060 dan peor rendimiento que con Ryzen 5600X
        all_e_eq = []
        with open(self.pkl_path, "rb") as f:
            pickle.load(f)  # Saltar el contador
            while True:
                try:
                    chunk = pickle.load(f)
                    while chunk:
                        all_e_eq.append(torch.tensor(chunk.pop(0)["equivalent_youngs_modulus"], dtype=torch.float))
                except EOFError:
                    break
        
#        stacked_all_e_eq = torch.cat(all_e_eq, dim=0)
        stacked_all_e_eq =torch.tensor(all_e_eq, dtype=torch.float)
        self.e_eq_min = stacked_all_e_eq.min(dim=0).values
        self.e_eq_max = stacked_all_e_eq.max(dim=0).values
        del all_e_eq, stacked_all_e_eq
        print_now("Finished _calculate_normalization")

    def _process_chunk(self, global_start: int, chunk_segment: list):
        processed_files=0
        for local_idx, lattice in enumerate(chunk_segment):
            idx = global_start + local_idx

            # Convertir a tensor y normalizar
            nodes = torch.tensor(lattice["nodes"], dtype=torch.float)
#            nodes_normalized = self._normalize_nodes(nodes)
            # node_features = torch.tensor(nodes_normalized, dtype=torch.float)
            edge_index = torch.tensor(lattice['edges'], dtype=torch.long).t().contiguous()

            edge_attrs = torch.tensor([
                list(nodes[i]) + list(nodes[j]) + [p['length']]
                for (i, j), p in zip(lattice['edges'], lattice['edge_properties'])
            ], dtype=torch.float)

            y = torch.tensor([
                lattice['equivalent_youngs_modulus'] / self.e_eq_max.item(),  # Normalizado correctamente
                lattice['relative_density']
            ], dtype=torch.float).unsqueeze(0)

            data = LatticeData(
                x= nodes,
                edge_index=edge_index,
                edge_attr=edge_attrs,
                y=y,
                ref_coords=nodes.clone()
            )

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            processed_files += 1
            del nodes,edge_attrs,edge_index,data,y
 #           del nodes_normalized
        return processed_files
    
    def get_normalization_params(self):
        """Devuelve los parámetros de normalización para desescalar"""
        return {'max': self.e_eq_max}
    
    def len(self) -> int:
#        return self.num_files
        return self.processed_files

    def get(self, idx: int) -> Data:
        # Cargar con weights_only=False para permitir objetos complejos
        data = torch.load(
            osp.join(self.processed_dir, f'data_{idx}.pt'),
            weights_only=False  # Solo si confías en el origen de los datos
        )
        return data

def custom_collate_fn(data_list):
    batch = Batch.from_data_list(data_list)
    batch.ref_coords = torch.cat([data.ref_coords for data in data_list], dim=0)
    return batch