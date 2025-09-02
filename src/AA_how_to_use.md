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

Características adicionales:

num_samples: Número de ejemplos a generar (por defecto: 50000).

min_e_eq: Límite inferior para el módulo de Young equivalente (por defecto: 80000000.00).

max_e_eq: Límite superior para el módulo de Young equivalente (por defecto: 120000000.00).

cpu_fraction: Fracción de potencia de CPU utilizada (0.0 a 1.0; por defecto: 0.4).

min_nodes: Número mínimo de nodos por lattice (por defecto: 10).

max_nodes: Número máximo de nodos por lattice (por defecto: 50).

update_batch: Número de lattices guardados en cada operación de escritura (por defecto: 50000).

--output_dir: Carpeta donde se guardarán los lattices generados (por defecto: data/lattices/).

--use_pickle: Si se debe usar pickle para guardar los lattices (por defecto: True).

--plot_lattice: Si se deben generar visualizaciones de los lattices (por defecto: False).

Ejemplo de ejecución completo:

__python Genlattice5.py 100000 80000000 120000000 0.4 10 50 50000 --output_dir data/lattices/ --use_pickle True --plot_lattice True__

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

Caracterísicas adicionales:

input_dir: Carpeta que contiene los archivos de datos originales (por ejemplo, lattices generados con Genlattice5.py).

output_dir: Carpeta donde se guardarán los datos procesados y escalados (por defecto: data/processed/).

train_frac: Fracción de datos que se asignará al conjunto de entrenamiento (por defecto: 0.8).

val_frac: Fracción de datos que se asignará al conjunto de validación (por defecto: 0.2).

scale_method: Método de escalado para normalizar los valores del dataset (minmax o standard; por defecto: minmax).

--shuffle: Si se deben mezclar los datos antes de realizar los splits (por defecto: True).

--save_numpy: Si se deben guardar los datasets procesados en formato .npy además del formato por defecto (por defecto: False).

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

Características adicionales:

Guarda el mejor modelo en best_decoder_dual.pth según el MSE promedio.

Registra métricas (MSE, MAE, porcentaje de nodos con error <2%) en un archivo CSV definido por METRIC_CSV.

Cada 10 épocas, genera visualizaciones de reconstrucción para los primeros 3 lattices del batch de prueba mediante plot_reconstruction.


Ejemplo de ejecución completo:

__python full_decoder_based.py --latent_dim 32 --batch_size 32 --num_epochs 301__

## z_generation.py

Script que entrena un generador de embeddings latentes (ZGenerator) a partir de ruido gaussiano. El generador produce vectores en el espacio latente, que se decodifican mediante un decoder previamente entrenado (best_decoder_dual.pth) para obtener coordenadas de nodos en 3D.

__python z_generation.py [--num_steps N] [--batch_size N]__

__1. --num_steps (opcional)__

Descripción: Número total de pasos de entrenamiento del generador.

Valor por defecto: 1000

__2. --batch_size (opcional)__

Descripción: Número de muestras de ruido generadas por iteración.

Valor por defecto: 64

Características adicionales

El decoder se carga desde best_decoder_dual.pth y permanece congelado durante el entrenamiento.

La función de pérdida penalizalas coordenadas fuera de los límites [0,1] y el colapsos de nodos (coincidencia excesiva en posiciones).

Se guardan imágenes (sample_XXXX.png) cada 50 pasos en la carpeta zgen_training_outputs/.

Se guarda el modelo entrenado como zgen_latest.pth cada 200 pasos.

Ejemplo de ejecución completo

__python z_generation.py --num_steps 1500 --batch_size 128__

z_modulator.py

Script que entrena un ZModulator, encargado de ajustar los vectores latentes generados (z) para que correspondan a un valor deseado de módulo de Young equivalente (E_eq).

Este módulo trabaja junto con:

Decoder (best_decoder_dual.pth) – congelado.

Predictor del encoder (best_encoder_predictor_nopool.pth) – congelado.

## Generador Z preentrenado (zgen_latest.pth).

__python z_modulator.py [--steps N] [--batch_size N]__

__1. --steps (opcional)__

Descripción: Número de pasos de entrenamiento del ZModulator.

Valor por defecto: 1000

__2. --batch_size (opcional)__

Descripción: Número de muestras de ruido generadas en cada paso de entrenamiento.

Valor por defecto: 64

Características adicionales

Entrenamiento supervisado indirecto:

Se generan z iniciales mediante un ZGenerator preentrenado.

El ZModulator ajusta z según un valor objetivo de E_eq en el rango [0,1].

El predictor compara el E_eq predicho con el deseado (loss = MSE).

Normalización de coordenadas: Las coordenadas reconstruidas se escalan a [0.05, 0.7] en cada eje para evitar colapsos en bordes.

Visualización:

Cada 50 pasos guarda un gráfico 3D (generated_XXXX.png) con los nodos reconstruidos y el valor de E_eq objetivo.

Guardado de modelos:

Cada 200 pasos guarda el estado del modulador en zmodulator_latest.pth.

Ejemplo de ejecución completo

__python z_modulator.py --steps 1500 --batch_size 128__

## generate_lattice_from_z.py

Script que genera geometrías de lattices en formato JSON a partir de valores deseados de módulo de Young equivalente (E_eq).

El flujo de trabajo es:

Se genera un vector latente (z) mediante un generador preentrenado.

El ZModulator ajusta z para acercarse al E_eq deseado.

El Decoder reconstruye las coordenadas 3D de los nodos.

Se calculan las conexiones mediante triangulación de Delaunay y se añaden los bordes del cubo.

Se predice el E_eq del lattice generado usando el Predictor.

Se guarda el resultado en un archivo .json.

Uso
__python generate_lattice_from_z.py --eq_values [E1 E2 ... En]__

Parámetros

__1. --eq_values (obligatorio)__

Descripción: Lista de valores objetivo de módulo de Young equivalente para los que se generarán lattices.

Ejemplo:

--eq_values 0.3 0.5 0.8

Características adicionales

Los modelos cargados (zgen, zmod, decoder, predictor) deben estar previamente entrenados:

zgen_latest.pth (ZGenerator).

zmodulator_latest.pth (ZModulator).

best_decoder_dual.pth (Decoder).

best_encoder_predictor_nopool.pth (Encoder+Predictor).

Las geometrías se escalan a un cubo interior de volumen 0.9x0.9x0.9 para evitar colapsos en los bordes.

Cada lattice se guarda en la carpeta generated_lattices_json/ como un archivo .json con el siguiente formato:

{
  "nodes": [[x, y, z], ...],
  "edges": [[i, j], ...],
  "equivalent_youngs_modulus": valor_predicho,
  "target_youngs_modulus": valor_objetivo
}

Ejemplo de ejecución
__python generate_lattice_from_z.py --eq_values 0.25 0.50 0.75__


__python generate_lattice_from_z.py --eq_values 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7__

## FEM_validator.py

Script que valida los lattices generados en formato JSON mediante simulaciones FEM. Calcula el módulo de Young equivalente (E_eq), lo compara con el valor objetivo y guarda los resultados en un archivo CSV.

Uso
python FEM_validator.py --folder PATH_TO_JSONS

Parámetros

__1. --folder (obligatorio)__

Descripción: Carpeta que contiene los archivos lattice_*.json generados previamente con generate_lattice_from_z.py.

Ejemplo:

--folder generated_lattices_json/

Características adicionales

Lee todos los archivos .json en la carpeta indicada.

Calcula el módulo de Young equivalente (E_eq) mediante LatticeGenerator.compute_equivalent_youngs_modulus().

Escala el valor objetivo (target_youngs_modulus) a las mismas unidades (multiplicando por 163.73e6).

Calcula error absoluto y relativo entre target y computed.

Imprime un resumen en consola con los 5 mejores resultados (menor error relativo).

Guarda todos los resultados en results.csv dentro de la misma carpeta.

Ejemplo de ejecución
__python FEM_validator.py --folder generated_lattices_json/__
