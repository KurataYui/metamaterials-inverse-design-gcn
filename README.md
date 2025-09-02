# Bachelor's Thesis – Graph Convolutional Autoencoder for Metamaterials

This repository contains the implementation developed for my Bachelor's Thesis on **inverse design of mechanical metamaterials**. The project leverages **graph neural networks (GNNs)** and a **Graph Convolutional Autoencoder (GCAE)** to encode lattice geometries, predict their equivalent Young’s modulus, and reconstruct node positions for geometry generation.

## Repository structure
- `src/` – source code with models, training scripts and utilities.
- `data/` – datasets of lattices and preprocessing utilities.
- `requirements.txt` – Python dependencies.
- `README.md` – this document.

## Features
- Data generation via **Delaunay-based lattice assembly** and FEM homogenization.
- GCAE architecture with **GCN/GAT encoders**, predictor, and node decoder.
- Training pipelines for property prediction, reconstruction, and inverse design.
- Latent space exploration with **conditional modulators**.
- Geometry generation in **JSON format** with predicted mechanical properties.

## Reference
This work was developed as part of my **Bachelor’s Thesis (2025)** at *Universidad Politécnica de Madrid (UPM)*.  
For full methodology, results, and figures, please refer to the accompanying thesis document.  

---
✨ Developed with passion for mechanical metamaterials and machine learning.
