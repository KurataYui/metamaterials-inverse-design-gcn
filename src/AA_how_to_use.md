# Guía de uso
Representación completa de uso del pipeline.
El orden de ejecución es el siguiente: 

1. genlattice5.py
2. processing_data.py  
3. full_model_fss_encoder_based.py
4. full_model_fss_decoder.py
5. diagnose_z_generation.py
6. train_zmodulator.py
7. generate_lattice_from_z.py
8. FEM_validator.py


## Genlattice5.py
Script que se ocupa de la generación de lattices, además de otras funcionalidades: 

__python genlattice5.py [num_samples] [min_e_eq] [max_e_eq] [cpu_fraction] [min_nodes] [max_nodes] [update_batch] [--output_dir DIR] [--use_pickle BOOL] 
[--plot_lattice BOOL]__

__1. num_samples (opcional)__
   
Descripción: Número de ejemplos a generar

Valor por defecto: 50000

__2. min_e_eq (opcional)__
   
Descripción: Límite inferior para el módulo de Young equivalente

Valor por defecto: 80000000.00

__3. max_e_eq (opcional)__

Descripción: Límite superior para el módulo de Young equivalente

Valor por defecto: 120000000.00

__4. cpu_fraction (opcional)__

Descripción: Fracción de potencia de CPU utilizada (0.0 a 1.0)

Valor por defecto: 0.4

__5. min_nodes (opcional)__

Descripción: Número mínimo de nodos por lattice

Valor por defecto: 10

__6. max_nodes (opcional)__
 
Descripción: Número máximo de nodos por lattice

Valor por defecto: 50

__7. update_batch (opcional)__
Descripción: Número de lattices guardados en cada operación de escritura

Valor por defecto: 50000

## Processing_data.py

Script que se encarga de la preparación y procesamiento de los datasets generados, incluyendo normalización, escalado y creación de splits para entrenamiento y validación.

__python processing_data.py [input_dir] [output_dir] [train_frac] [val_frac] [scale_method] [--shuffle BOOL] [--save_numpy BOOL]__

__1. input_dir (obligatorio)__

Descripción: Carpeta que contiene los archivos de datos originales (por ejemplo, lattices generados con genlattice5.py).

Valor por defecto: Ninguno (requerido)

Ejemplo (usado en el trabajo): data/raw_lattices/ 

__2. output_dir (opcional)__

Descripción: Carpeta donde se guardarán los datos procesados y escalados.

Valor por defecto: data/processed/

__3. train_frac (opcional)__

Descripción: Fracción de datos que se asignará al conjunto de entrenamiento.

Valor por defecto: 0.8

__4. val_frac (opcional)__

Descripción: Fracción de datos que se asignará al conjunto de validación. El resto se puede usar para test.

Valor por defecto: 0.2

__5. scale_method (opcional)__

Descripción: Método de escalado para normalizar los valores del dataset. Opciones disponibles:

minmax: escala los valores entre 0 y 1.

standard: normaliza los valores con media 0 y desviación estándar 1.

Valor por defecto: minmax

__6. --shuffle (opcional)__

Descripción: Indica si se deben mezclar los datos antes de realizar los splits.

Valor por defecto: True

Ejemplo: --shuffle False

__7. --save_numpy (opcional)__

Descripción: Guarda los datasets procesados en formato .npy además del formato por defecto (por ejemplo, .csv).

Valor por defecto: False

Ejemplo: --save_numpy True

Ejemplo de ejecución completo: 

__python processing_data.py data/raw_lattices/ data/processed/ 0.8 0.2 minmax --shuffle True --save_numpy True__

Recomendación: variar los parámetros en el script y ejecutar solo con __python processing_data.py__

## Full_model_fss_encoder_based.py

Entrenamiento del encoder basado en GAT (Graph Attention Network) para la generación de representaciones latentes de estructuras de lattice.

__python full_model_fss_encoder_based.py [--epochs EPOCHS] [--batch_size BATCH_SIZE]__

#### Para cambiar la dimensión latente cambiar latent_dim en el script

__1. --epochs (opcional)__

Descripción: Número de épocas de entrenamiento del modelo.

Valor por defecto: 201

Ejemplo: --epochs 300

__2. --batch_size (opcional)__

Descripción: Tamaño de cada batch de entrenamiento.

Valor por defecto: 32

Ejemplo: --batch_size 64

Ejemplo de ejecución completo:

__python full_encoder_based.py --epochs 250 --batch_size 64__

## Full_model_fss_decoder.py

Script que entrena el decoder de nodos para reconstruir las coordenadas 3D de los lattices a partir del espacio latente generado por el encoder. Utiliza un encoder previamente entrenado (best_encoder_predictor_nopool.pth) congelado durante el entrenamiento.

__python full_decoder_based.py [--latent_dim N] [--batch_size N] [--num_epochs N]__

__1. --latent_dim (opcional)__

Descripción: Dimensión del espacio latente que recibe el decoder como entrada.

Valor por defecto: 32
_IMPORTANTE: usar el mismo valor que el encoder_

__2. --batch_size (opcional)__

Descripción: Número de lattices procesados en cada batch durante el entrenamiento.

Valor por defecto: 32
_IMPORTANTE: usar el mismo valor que el encoder_

__3. --num_epochs (opcional)__

Descripción: Número de épocas de entrenamiento del decoder.

Valor por defecto: 301

Características adicionales

Guarda el mejor modelo en best_decoder_dual.pth según el MSE promedio.

Registra métricas (MSE, MAE, porcentaje de nodos con error <2%) en un archivo CSV definido por METRIC_CSV.

Cada 10 épocas, genera visualizaciones de reconstrucción para los primeros 3 lattices del batch de prueba mediante plot_reconstruction.


Ejemplo de ejecución completo:

__python full_decoder_based.py --latent_dim 32 --batch_size 32 --num_epochs 301__


