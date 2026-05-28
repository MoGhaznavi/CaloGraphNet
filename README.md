
# CaloGraphNet

> **Graph Neural Networks for Calorimeter Cell Clustering in Particle Physics**
> 
> Processes ATLAS ROOT files into graph datasets and trains GNNs (GCN, GAT, Graph Transformer, SAGE) for edge classification.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Pipeline Architecture](#pipeline-architecture)
- [File 1: build_graph_dataset.py](#file-1-build_graph_datasetpy)
  - [What it does](#what-it-does)
  - [The 4 Output Files](#the-4-output-files)
  - [Geometry Handling](#geometry-handling)
  - [Label Logic](#label-logic)
  - [Usage](#usage-build_graph_datasetpy)
- [File 2: train_gnn_models.py](#file-2-train_gnn_modelspy)
  - [What it does](#what-it-does-1)
  - [Feature Sets](#feature-sets)
  - [Model Architectures](#model-architectures)
  - [Loss Functions](#loss-functions)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Output Files](#output-files)
  - [Usage](#usage-train_gnn_modelspy)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Citing This Work](#citing-this-work)
- [License](#license)

---

## Overview

CaloGraphNet is a complete pipeline for graph-based machine learning on calorimeter cell data from particle physics detectors (e.g., ATLAS). It:

1. **Converts ROOT files** → Graph-structured datasets (HDF5 + NumPy)
2. **Trains Graph Neural Networks** for edge classification
3. **Supports multiple architectures**: GCN, GAT, Graph Transformer, GraphSAGE

---

## Problem Statement

In a calorimeter, particles deposit energy across multiple cells. The goal is to group cells belonging to the same physical particle (clustering). We formulate this as an **edge classification problem** on a graph where:

- **Nodes** = Calorimeter cells
- **Edges** = Geometric neighbor connections between cells

For each edge (cell pair), the model predicts one of **5 classes**:

| Class | Label | Meaning |
|-------|-------|---------|
| 0 | Noise-Noise | Both cells are noise (no particle cluster) |
| 1 | Same Cluster | Both cells belong to the same particle cluster |
| 2 | Source Only | Source cell in cluster, destination is noise |
| 3 | Destination Only | Destination in cluster, source is noise |
| 4 | Different Clusters | Cells belong to different particle clusters |

**Why edge classification?** Once we predict which cell pairs belong to the same cluster, we can use graph clustering algorithms (e.g., connected components) to reconstruct full particle clusters.

---

## Pipeline Architecture

```
ROOT File (ROD format from ATLAS)
       ↓
╔══════════════════════════════════════════════════════════════════╗
║              build_graph_dataset.py                              ║
║  - Reads cell-level branches (energy, noise, eta, phi, etc.)    ║
║  - Builds graph connectivity from neighbor information          ║
║  - Defines geometry via CELL_SIZES mapping                       ║
║  - Computes per-cell noise statistics across all events         ║
║  - Saves 4 output files                                          ║
╚══════════════════════════════════════════════════════════════════╝
       ↓
  HDF5 + NumPy Dataset (4 files)
       ↓
╔══════════════════════════════════════════════════════════════════╗
║              train_gnn_models.py                                 ║
║  - Loads dataset with feature selection (baseline vs all)       ║
║  - Builds GNN model (GCN, GAT, Transformer, or SAGE)           ║
║  - Trains with weighted loss for class imbalance                ║
║  - Evaluates with comprehensive metrics (F1 Sum Score, etc.)    ║
║  - Saves model checkpoints and predictions (Parquet format)     ║
╚══════════════════════════════════════════════════════════════════╝
       ↓
  Trained Model + Predictions + Metrics
```

---

## File 1: build_graph_dataset.py

### What it does

This script takes a ROOT file containing calorimeter cell data and converts it into a graph-structured dataset suitable for GNN training. It processes the data in chunks to handle large files efficiently.

**Key operations:**
1. Reads cell-level branches from the ROOT file (energy, noise, eta, phi, cluster indices, neighbor connectivity)
2. Filters out invalid cells (e.g., where noise = 0)
3. Builds a graph by creating undirected edges between neighboring cells (using the `neighbor` branch)
4. Computes static cell metadata (subcalorimeter type, sampling layer, cell dimensions, volume)
5. Computes per-cell noise statistics (mean, std, category) across all events
6. Fits an energy scaler (robust or standard normalization) for consistent feature scaling
7. Processes each event to extract energy/SNR values and cluster assignments
8. Assigns edge labels based on cluster membership (5-class scheme)
9. Saves everything to 4 structured output files

### The 4 Output Files

All outputs are saved in: `output_dir/{dataset_name}/`

#### 1. `pairs_{dataset_name}.npy`
- **Format**: NumPy array, shape `(num_edges, 2)`, dtype `int32`
- **What it contains**: The graph connectivity - each row is `[source_cell_idx, target_cell_idx]`
- **How it's built**: From the `neighbor` branch in the ROOT file, which lists neighboring cells for each cell. The script:
  - Maps original cell indices to global indices (after filtering invalid cells)
  - Creates undirected edges (ensures src < dst)
  - Removes duplicate edges
  - Orders edges by source cell for efficient lookup
- **Size**: Typically millions to hundreds of millions of edges

#### 2. `cells_{dataset_name}.npy`
- **Format**: Structured NumPy array, shape `(num_cells,)` with named fields
- **What it contains**: Static metadata for each cell (does NOT change per event):

| Field | Type | Description |
|-------|------|-------------|
| `global_idx` | int32 | Index in this dataset (0 to num_cells-1) |
| `orig_idx` | int32 | Original index from ROOT file |
| `eta_event0` | float64 | Eta coordinate from first event (used as static geometry) |
| `phi_event0` | float64 | Phi coordinate from first event |
| `broken_event0` | bool | Whether cell had noise=0 in event 0 |
| `num_neighbors` | int32 | Number of neighboring cells (degree) |
| `subcalo` | int32 | Sub-calorimeter ID (0=EMB, 1=EMEC, 2=HEC, 3=Tile) |
| `sampling` | int32 | Sampling layer index |
| `deta` | float32 | Cell size in eta (radians) - from CELL_SIZES mapping |
| `dphi` | float32 | Cell size in phi (radians) - from CELL_SIZES mapping |
| `volume` | float32 | Cell volume (deta × dphi × R, where R=1000mm) |
| `noise_mean` | float32 | Mean noise across all events for this cell |
| `noise_std` | float32 | Standard deviation of noise across all events |
| `noise_count` | int32 | Number of events this cell had valid noise |
| `noise_category` | int8 | Categorization: 0=low noise, 1=medium, 2=high, -1=invalid |

- **Why it's useful**: Provides static features that don't change per event, saving disk space and allowing reuse across multiple training runs.

#### 3. `events_{dataset_name}.h5`
- **Format**: HDF5 file with multiple datasets, shape `(num_events, num_cells)` for each dataset
- **What it contains**: Per-event, per-cell features (dynamic data):

| Dataset | Description |
|---------|-------------|
| `cell/energy_raw` | Raw energy deposits (if available) |
| `cell/noise_raw` | Raw noise values (if available) |
| `cell/snr_computed` | Signal-to-noise = energy / noise (computed) |
| `cell/snr_raw` | Direct SNR from ROOT (if available) |
| `cell/energy_normalized` | Normalized energy: `(energy - center) / scale` |
| `cell/cell_cluster_index` | Cluster ID for each cell (0=noise, ≥1=valid cluster) |

- **HDF5 attributes** (metadata stored in file):
  - `dataset_name`, `input_file`, `num_events`, `num_cells`, `num_pairs`

#### 4. `labels_{dataset_name}.npy`
- **Format**: NumPy array, shape `(num_events, num_edges)`, dtype `int8`
- **What it contains**: Ground truth labels for each edge in each event
- **Label assignment logic** (critical for correct training):

```python
if c0 == 0 and c1 == 0:      # Both noise
    label = 0
elif c0 == c1 and c0 >= 1:   # Same valid cluster
    label = 1
elif c0 >= 1 and c1 == 0:    # Source in cluster, destination noise
    label = 2
elif c0 == 0 and c1 >= 1:    # Source noise, destination in cluster
    label = 3
elif c0 != c1 and c0 >= 1 and c1 >= 1:  # Different clusters
    label = 4
```

### Geometry Handling

**Where geometry is defined** (lines ~169-185):

```python
CELL_SIZES = {
    '0_0': (0.025, 0.1),     # (Δη, Δφ) for subcalo 0, sampling 0
    '0_1': (0.0031, 0.0245), # Electromagnetic barrel, fine sampling
    '0_2': (0.025, 0.0245),
    '0_3': (0.05, 0.0245),
    '0_4': (0.025, 0.1),
    '1_8': (0.1, 0.09817481), # EMEC (endcap)
    '2_21': (0.1, 0.1),       # HEC (hadronic)
    '3_12': (0.1, 0.09817481), # Tile (hadronic barrel)
    # ... etc.
}
```

This mapping connects `(subCalo, sampling)` pairs to physical cell dimensions in η-φ space. The mapping was derived from the ATLAS detector geometry.

**Geometry features computed per cell** (lines ~300-330):
- `eta`, `phi`: Cell center coordinates from ROOT branches
- `deta`, `dphi`: Cell dimensions from CELL_SIZES mapping
- `volume`: `deta × dphi × 1000 mm` (approximate)
- `subcalo`, `sampling`: One-hot encoded for model input

### Usage: build_graph_dataset.py

```bash
python build_graph_dataset.py \
    --input /path/to/file.root \
    --output-dir /storage/processed_data \
    --dataset-name my_dataset \
    --normalize-energy \
    --energy-normalization robust \
    --chunk-size 50 \
    --batch-size 20
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | Required | Input ROOT file path |
| `--output-dir`, `-o` | `/storage/mxg1065/processed_data` | Base output directory |
| `--dataset-name`, `-n` | Auto from filename | Custom dataset name |
| `--normalize-energy` | True | Apply energy normalization |
| `--energy-normalization` | `robust` | `robust` (median/IQR) or `standard` |
| `--skip-snr-scaling` | False | Don't scale SNR features |
| `--batch-size` | 20 | Events per batch |
| `--chunk-size` | 50 | Events per chunk (memory control) |
| `--debug` | True | Enable debug output |

---

## File 2: train_gnn_models.py

### What it does

This script loads the dataset created by `build_graph_dataset.py`, trains Graph Neural Networks for edge classification, and saves comprehensive results.

**Key operations:**
1. Loads features with user selection (baseline 3-feature or all 42+ features)
2. Splits events into train/test sets (default 70/30)
3. Builds a GNN model (GCN, GAT, Graph Transformer, or GraphSAGE)
4. Sets up loss function (standard, weighted, or focal loss for class imbalance)
5. Trains with mixed precision (FP16) for faster GPU training
6. Evaluates using comprehensive metrics including F1 Sum Score (class imbalance-aware)
7. Saves best model checkpoint and prediction results (Parquet format)

### Feature Sets

#### Baseline Mode (`--baseline`) - 3 features
| Feature | Source | Description |
|---------|--------|-------------|
| SNR | `events.h5` | Signal-to-noise ratio (energy/noise) |
| η (eta) | `events.h5` | Pseudo-rapidity coordinate |
| φ (phi) | `events.h5` | Azimuthal angle coordinate |

**Use baseline when:** You want a fast, minimal model for quick experiments or as a reference baseline.

#### All Features Mode (`--all-features`) - 42+ features
| Category | Features | Source |
|----------|----------|--------|
| Energy/SNR | SNR (computed), raw energy, raw noise | `events.h5` |
| Geometry | η, φ, Δη, Δφ, volume | `events.h5` + `cells.npy` |
| Noise Stats | noise_mean, noise_std, noise_count, noise_category | `cells.npy` |
| Topology | num_neighbors | `cells.npy` |
| Detector Info | subcalo one-hot (4 classes), sampling one-hot (3+ classes) | `cells.npy` |
| Cluster Info | in_cluster flag, cluster_id_norm | `events.h5` |

**Use all features when:** You want the best possible performance and have sufficient GPU memory.

### Model Architectures

All models follow the same unified design:

```
Input: Node features (N_cells × F)
       ↓
Node Embedding (Linear: F → hidden_dim)
       ↓
L GNN Layers (configurable)
  ├── Message passing (GCNConv / GATConv / TransformerConv / SAGEConv)
  ├── BatchNorm / LayerNorm
  ├── ReLU activation
  └── Residual connection
       ↓
Edge Representation: Concatenate [h_src, h_dst] for each edge
       ↓
Classifier (Linear: 2*hidden_dim → 5)
       ↓
Output: Logits for 5 classes
```

| Model | `--model` | Key Parameter | Best For |
|-------|-----------|---------------|----------|
| GCN | `gcn` | Symmetric normalization | Baseline, fast training |
| GAT | `gat` | Multi-head attention | Graphs with varying importance |
| Graph Transformer | `transformer` | Self-attention on edges | Long-range dependencies |
| GraphSAGE | `sage` | Neighborhood aggregation | Large graphs, inductive learning |

### Loss Functions

#### Focal Loss (Recommended for extreme class imbalance)
```python
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```
- **Class-specific alphas** (hardcoded for physics dataset):
  ```python
  RECOMMENDED_ALPHAS = [0.10, 0.60, 0.70, 0.70, 1.00]  # Classes 0,1,2,3,4
  ```
- **γ (gamma)** = 2.0 (controls focus on hard examples)
- Use with: `--weighted-loss --weight-strategy focal`

#### Weighted CrossEntropy
- Weighted by inverse class frequency
- Use with: `--weighted-loss --weight-strategy inverse`

#### Standard CrossEntropy
- No class weighting (default)

### Evaluation Metrics

The script computes comprehensive metrics. **Best model selection uses F1 Sum Score** (accounts for both false positives and false negatives).

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **F1 Sum Score (FSS)** | Σ(class F1) | 1.0 = random, 5.0 = perfect |
| **Recall Sum Score (RSS)** | Σ(class recall) | 1.0 = random, 5.0 = perfect |
| **Macro F1** | Mean(class F1) | Each class equally important |
| **Weighted F1** | Weighted by class frequency | Favors frequent classes |
| **Per-class metrics** | Precision, recall, F1 | Debug individual class performance |

**Why F1 Sum Score?** Traditional accuracy is misleading with class imbalance (e.g., if 90% of edges are class 0, a model predicting all class 0 gets 90% accuracy but is useless). FSS penalizes models that ignore rare but important classes.

### Output Files

Training produces these files in `--save-dir` (default: `/storage/mxg1065/foundation_experiments/`):

#### 1. Model Checkpoints
- `best_{exp_name}.pt` - Best model based on F1 Sum Score
- `{exp_name}_epoch{N}.pt` - Checkpoint at each epoch (for resuming)

#### 2. Metrics File
- `{exp_name}_metrics.pkl` - Pickle file containing:
  - Training history (loss, accuracy per epoch)
  - Test metrics per epoch
  - Best epoch and best metrics
  - Model arguments (for reproducibility)

#### 3. Predictions (Parquet format)
- `results_{exp_name}.parquet` - Per-edge predictions with schema:

| Column | Type | Description |
|--------|------|-------------|
| `event_id` | int32 | Event index |
| `edge_id` | int32 | Edge index within event |
| `source_id` | int32 | Source cell global index |
| `target_id` | int32 | Target cell global index |
| `true_label` | int8 | Ground truth (0-4) |
| `pred_label` | int8 | Model prediction (0-4) |
| `confidence` | float32 | Softmax probability of predicted class |
| `score_class_0` | float32 | Raw logit / probability for class 0 |
| `score_class_1` | float32 | Raw logit / probability for class 1 |
| `score_class_2` | float32 | Raw logit / probability for class 2 |
| `score_class_3` | float32 | Raw logit / probability for class 3 |
| `score_class_4` | float32 | Raw logit / probability for class 4 |
| `model_name` | string | Model identifier |
| `source_cluster` | int32 | Cluster ID of source cell (if available) |
| `target_cluster` | int32 | Cluster ID of target cell (if available) |
| `same_cluster` | bool | Whether cells share same cluster (if available) |

**Parquet advantages:** Columnar format, fast to query, efficient compression (ZSTD), works with Pandas/Dask/Spark.

### Usage: train_gnn_models.py

```bash
# Train baseline GCN
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --data-dir /storage/processed_data/my_dataset \
    --gpu 0 \
    --epochs 30

# Train Graph Transformer with all features and focal loss
python train_gnn_models.py \
    --model transformer \
    --all-features \
    --hidden-dim 256 \
    --layers 8 \
    --heads 4 \
    --weighted-loss \
    --weight-strategy focal \
    --focal-gamma 2.0 \
    --gpu 1

# Train all architectures sequentially (for comparison)
python train_gnn_models.py \
    --model all \
    --baseline \
    --epochs 20 \
    --gpu 0

# Run inference only (use best checkpoint, no training)
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --inference-only \
    --data-dir /storage/processed_data/my_dataset \
    --gpu 0
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model`, `-m` | `gcn` | Model type: `gcn`, `gat`, `transformer`, `sage`, `all` |
| `--baseline` | True | Use 3 features only |
| `--all-features` | False | Use all 42+ features |
| `--hidden-dim` | 128 | Hidden dimension size |
| `--layers`, `-l` | 6 | Number of GNN layers |
| `--heads` | 2 | Attention heads (GAT/Transformer) |
| `--dropout` | 0.0 | Dropout rate |
| `--norm` | `batch` | Normalization: `batch`, `layer`, `none` |
| `--epochs`, `-e` | 30 | Training epochs |
| `--batch-size`, `-b` | 1 | Batch size (events per batch) |
| `--lr` | 1e-3 | Learning rate |
| `--weighted-loss` | False | Use weighted loss for imbalance |
| `--weight-strategy` | `inverse` | `inverse`, `focal`, `logarithmic` |
| `--focal-gamma` | 2.0 | Focal loss focusing parameter |
| `--focal-alpha` | 0.25 | Alpha parameter for focal loss |
| `--gpu`, `-g` | 0 | GPU ID (-1 for CPU) |
| `--mixed-precision` | True | Enable FP16 training (faster on GPU) |
| `--inference-only` | False | Skip training, only run inference |
| `--resume` | True | Resume from checkpoint if exists |
| `--patience` | 10 | Early stopping patience |

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/CaloGraphNet.git
cd CaloGraphNet

# Install dependencies
pip install torch torch-geometric numpy awkward uproot h5py pandas pyarrow scikit-learn psutil

# Optional: For progress bars in inference
pip install tqdm
```

### Complete Example: From ROOT to Trained Model

```bash
# Step 1: Build dataset from ROOT file
python build_graph_dataset.py \
    --input /data/run12345.root \
    --output-dir /my_data \
    --dataset-name physics_run \
    --normalize-energy

# Output will be in: /my_data/physics_run/

# Step 2: Train a GCN model
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --data-dir /my_data/physics_run \
    --gpu 0 \
    --epochs 30

# Step 3: (Optional) Run inference on best model
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --inference-only \
    --data-dir /my_data/physics_run \
    --gpu 0

# Results are in: /storage/mxg1065/foundation_experiments/
# - best_gcn_baseline_h128_l6.pt (model)
# - results_gcn_baseline_h128_l6.parquet (predictions)
```

### Analyze Results with Pandas

```python
import pandas as pd

# Load predictions
df = pd.read_parquet("results_gcn_baseline_h128_l6.parquet")

# Overall accuracy
accuracy = (df['true_label'] == df['pred_label']).mean()
print(f"Accuracy: {accuracy:.4f}")

# Per-class F1 (requires grouping)
def f1_per_class(df):
    results = {}
    for c in range(5):
        tp = ((df['true_label'] == c) & (df['pred_label'] == c)).sum()
        fp = ((df['true_label'] != c) & (df['pred_label'] == c)).sum()
        fn = ((df['true_label'] == c) & (df['pred_label'] != c)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results[c] = {'precision': precision, 'recall': recall, 'f1': f1}
    return results

f1_scores = f1_per_class(df)
print(f"F1 Sum Score: {sum(v['f1'] for v in f1_scores.values()):.2f}")
```

---

## Configuration Reference

### build_graph_dataset.py Full Arguments

```
usage: build_graph_dataset.py [-h] --input INPUT [--output-dir OUTPUT_DIR]
                              [--dataset-name DATASET_NAME] [--debug]
                              [--skip-snr-scaling] [--normalize-energy]
                              [--energy-normalization {robust,standard}]
                              [--workers WORKERS] [--batch-size BATCH_SIZE]
                              [--chunk-size CHUNK_SIZE]
                              [--h5-compression H5_COMPRESSION]
                              [--h5-chunk-rows H5_CHUNK_ROWS]

Required:
  --input, -i INPUT       Input ROOT file path

Optional:
  --output-dir, -o DIR    Base output directory
  --dataset-name, -n NAME Custom dataset name
  --debug                 Enable debug output (default: True)
  --skip-snr-scaling      Don't scale SNR features
  --normalize-energy      Apply energy normalization (default: True)
  --energy-normalization  'robust' (median/IQR) or 'standard'
  --batch-size N          Events per batch (default: 20)
  --chunk-size N          Events per chunk (default: 50)
  --h5-compression        'gzip', 'lzf', or None (default: 'gzip')
```

### train_gnn_models.py Full Arguments

```
usage: train_gnn_models.py [-h] [--model {gcn,gat,transformer,sage,all}]
                           [--hidden-dim HIDDEN_DIM] [--layers LAYERS]
                           [--heads HEADS] [--dropout DROPOUT]
                           [--layer-weights] [--softmax-weights]
                           [--norm {batch,layer,none}] [--baseline]
                           [--all-features] [--epochs EPOCHS]
                           [--batch-size BATCH_SIZE] [--lr LR]
                           [--weight-decay WEIGHT_DECAY] [--patience PATIENCE]
                           [--weighted-loss]
                           [--weight-strategy {inverse,focal,logarithmic,manual}]
                           [--focal-alpha FOCAL_ALPHA]
                           [--focal-gamma FOCAL_GAMMA] [--train-ratio TRAIN_RATIO]
                           [--gpu GPU] [--mixed-precision] [--no-mixed-precision]
                           [--save-dir SAVE_DIR] [--exp-name EXP_NAME]
                           [--data-dir DATA_DIR] [--resume] [--debug]
                           [--inference-only]

Model Architecture:
  --model {gcn,gat,transformer,sage,all}
                        Backbone architecture (default: gcn)
  --hidden-dim HIDDEN_DIM
                        Hidden dimension (default: 128)
  --layers, -l LAYERS   Number of GNN layers (default: 6)
  --heads HEADS         Attention heads for GAT/Transformer (default: 2)
  --dropout DROPOUT     Dropout rate (default: 0.0)
  --norm {batch,layer,none}
                        Normalization type (default: batch)

Feature Selection:
  --baseline            Use 3 baseline features (default: True)
  --all-features        Use all 42+ features

Training:
  --epochs, -e EPOCHS   Number of epochs (default: 30)
  --batch-size, -b BATCH_SIZE
                        Batch size in events (default: 1)
  --lr LR               Learning rate (default: 1e-3)
  --patience PATIENCE   Early stopping patience (default: 10)

Loss Functions:
  --weighted-loss       Use weighted loss for class imbalance
  --weight-strategy {inverse,focal,logarithmic,manual}
                        Class weighting strategy (default: inverse)
  --focal-gamma GAMMA   Focal loss gamma (default: 2.0)

Hardware:
  --gpu, -g GPU         GPU ID (-1 for CPU) (default: 0)
  --mixed-precision     Enable mixed precision (default: True)
  --no-mixed-precision  Disable mixed precision

Experiment:
  --save-dir SAVE_DIR   Directory to save models (default: /storage/...)
  --exp-name EXP_NAME   Experiment name (auto-generated if not provided)
  --data-dir DATA_DIR   Data directory (default: /storage/mxg1065/datafiles)
  --inference-only      Skip training, only run inference
  --resume              Resume from checkpoint (default: True)
  --debug               Debug mode (only runs a few events)
```
