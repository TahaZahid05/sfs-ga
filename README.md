# SfS-GA: Reconstructing Axially Symmetric Pottery Using a Genetic Algorithm

This repository contains the official C++ implementation of **SfS-GA**, a genetic algorithm-based system for the 3D reassembly of axially symmetric archaeological pottery from fractured sherds.

---

## Overview

The reassembly of broken archaeological pottery is a challenging combinatorial problem. Building on the [Structure-from-Sherds++ (SfS++)](https://arxiv.org/abs/2502.13986) pipeline, **SfS-GA** replaces incremental greedy beam search with a population-based evolutionary optimization framework.

Key features include:
- **Integer Chromosome Encoding:** Encodes candidate pairwise edge matches with specialized spanning tree repair via Disjoint Set Union (Kruskal's algorithm).
- **Domain-Specific Genetic Operators:** BFS-Partition crossover to preserve connected geometric sub-assemblies, localized edge and triad mutations, and a hyper-mutation / mass extinction mechanism to escape deceptive local minima.
- **Multi-Component Fitness Evaluation:** Incorporates LCS inlier density scores, connected component penalties, and voxel-based 3D collision/overlap detection.
- **Interactive PCL Visualization:** Inspect assembled pottery models with real-time toggling against ground truth.

---

## Quickstart (Docker with GUI)

The easiest and recommended way to run **SfS-GA** is using our provided Docker setup and helper shell scripts, which come with all dependencies (PCL, Ceres Solver, Eigen3, and OpenMP) pre-configured and include automatic X11 GUI forwarding for 3D visualization.

### 1. Download Dataset

Download and extract the standard SfS++ benchmark pottery dataset (Pots A through J):

```bash
chmod +x download.sh
./download.sh
```

### 2. Build the Docker Image

Build the container image tagged as `sfs:latest`:

```bash
docker build -t sfs:latest .
```

### 3. Launch Container with GUI Support

Run `setup_container.sh` to start an interactive container with your host X11 display mapped for GUI rendering:

```bash
chmod +x setup_container.sh
./setup_container.sh
```

### 4. Build and Run Inside Container

Once inside the container shell:

```bash
# Build the project
chmod +x build_at_container.sh
./build_at_container.sh

# Run reconstruction with interactive 3D viewer
cd build
./Hierarchy-Clear --icp-mode=global
```

---

## Interactive 3D Viewer Controls

When the reconstruction finishes, the interactive PCL 3D visualizer will open on your screen:
- Press **`G`** to toggle between the **GA Assembly** and **Archaeological Ground Truth**.
- Press **`Q`** to close the viewer.

---

## License

This project is released under the [LICENSE](LICENSE) provided in this repository.
