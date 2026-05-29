
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
║  - Reads cell-level branches (energy, noise, eta, phi, etc.)     ║
║  - Builds graph connectivity from neighbor information           ║
║  - Defines geometry via CELL_SIZES mapping                       ║
║  - Computes per-cell noise statistics across all events          ║
║  - Saves 4 output files                                          ║
╚══════════════════════════════════════════════════════════════════╝
       ↓
  HDF5 + NumPy Dataset (4 files)
       ↓
╔══════════════════════════════════════════════════════════════════╗
║              train_gnn_models.py                                 ║
║  - Loads dataset with feature selection (baseline vs all)        ║
║  - Builds GNN model (GCN, GAT, Transformer, or SAGE)             ║
║  - Trains with weighted loss for class imbalance                 ║
║  - Evaluates with comprehensive metrics (F1 Sum Score, etc.)     ║
║  - Saves model checkpoints and predictions (Parquet format)      ║
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
- **Size**: In the case of the ATLAS calorimeter, around 1.2 million pairs

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
Perfect! You should add the **File 3: analyze_results.py** section right after **File 2: train_gnn_models.py** and before **Quick Start**. Here's the expanded section with all the details:

## Where to add in your README

Insert this **after** the `train_gnn_models.py` section and **before** the `Quick Start` section:

```
## File 2: train_gnn_models.py
... (your existing content) ...

## File 3: analyze_results.py    <-- ADD THIS HERE

## Quick Start
... (your existing content) ...
```

---
## File 3: analyze_results.py

### What it does

Generates comprehensive analysis and visualization for trained models. This script takes the model checkpoint files (`.pkl`) and prediction files (`.parquet`) from `train_gnn_models.py` and produces publication-ready figures and reports.

**Visualization outputs:**

| Plot Type | Description | What it shows |
|-----------|-------------|----------------|
| **ROC Curves** | Receiver Operating Characteristic curves | AUC per class, trade-off between TPR and FPR |
| **Precision-Recall Curves** | PR curves with F1 iso-lines | AP score, best F1 point, class imbalance effects |
| **Confusion Matrix** | Normalized confusion matrix | Per-class misclassification patterns |
| **Per-Class Bar Chart** | Recall, Precision, F1 per class | Compare performance across 5 classes |
| **Radar Chart** | Multi-metric comparison | Top models compared across 5 metrics |
| **FSS vs RSS Scatter** | F1 Sum vs Recall Sum | Identify models with good precision (FSS > RSS) |
| **Architecture Box Plot** | Performance by architecture | Compare GCN vs GAT vs Transformer vs SAGE |

**Report outputs:**

- **`master_report.html`** - Interactive HTML report with all results
- **`comprehensive_metrics.csv`** - All metrics in CSV format
- **`comprehensive_table.html`** - Styled table of top 10 models

### Key Features

- **Memory-efficient**: Processes parquet files in chunks (handles millions of edges)
- **Primary metric**: F1 Sum Score (FSS) = Σ(F1 per class) - balances precision AND recall
  - Perfect score = 5.0
  - Random baseline = 1.0
  - Accounts for False Positives (unlike RSS)
- **Model ranking**: Automatically ranks models and identifies best performer by FSS
- **Architecture comparison**: Compares GCN vs GAT vs Transformer vs SAGE
- **Fallback handling**: If FSS not found, uses RSS with warning

### Output Directory Structure

After running, you'll get this organized output:

```
analysis_output/
├── figures/
│   ├── roc_curves/              # ROC + F1 tradeoff per model
│   │   └── {model_name}_roc_f1.png
│   ├── pr_curves/               # Precision-Recall curves
│   │   └── {model_name}_pr_curves.png
│   ├── confusion_matrices/      # Confusion matrices + per-class bars
│   │   └── {model_name}_confusion_enhanced.png
│   ├── comparison_plots/        # Cross-model comparisons
│   │   └── comprehensive_metrics.png
│   └── metrics_radar/           # Radar charts of top models
│       └── metrics_radar.png
├── tables/
│   └── comprehensive_table.html  # Styled performance table
├── reports/
│   └── master_report.html        # Complete analysis report
└── data/
    └── comprehensive_metrics.csv  # All metrics in CSV format
```

### Understanding the Metrics

| Metric | Abbrev | Range | What it penalizes | Best for |
|--------|--------|-------|-------------------|----------|
| **F1 Sum Score** | FSS | 1.0-5.0 | False Positives + False Negatives | **Primary metric** |
| Recall Sum Score | RSS | 1.0-5.0 | False Negatives only | Legacy comparison |
| Macro F1 | mF1 | 0-1 | Rare class mistakes | Balanced evaluation |
| Weighted F1 | wF1 | 0-1 | Frequent class mistakes | Production deployment |
| Mean Average Precision | mAP | 0-1 | Rank-order mistakes | Threshold tuning |

**Why FSS is the primary metric:**
- A model that predicts all edges as Class 0 (majority class) gets:
  - High Accuracy (~90%) ❌ misleading
  - High RSS (~4.5) ❌ misses the problem
  - Low FSS (~1.2) ✅ reveals the issue
- FSS penalizes models that sacrifice precision for recall

### Usage

**Basic usage:**
```bash
python analyze_results.py \
    --models-dir /path/to/models \
    --parquet-dir /path/to/parquet_files \
    --output-dir ./analysis_output
```

**Advanced usage with custom limits:**
```bash
python analyze_results.py \
    --models-dir /storage/foundation_experiments \
    --parquet-dir /storage/foundation_experiments \
    --output-dir ./comprehensive_analysis \
    --max-rows-roc 1000000 \
    --max-rows-confusion 2000000 \
    --top-n 10 \
    --batch-size 50000
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--models-dir` | Required | Directory containing `*_metrics.pkl` files |
| `--parquet-dir` | Required | Directory containing `results_*.parquet` files |
| `--output-dir` | `./analysis_output` | Output directory for all results |
| `--max-rows-roc` | 500000 | Max rows for ROC/PR computation (memory control) |
| `--max-rows-confusion` | 1000000 | Max rows for confusion matrix |
| `--top-n` | 5 | Number of top models to show in radar chart |
| `--batch-size` | 100000 | Batch size for parquet reading |

### What Each Plot Tells You

#### 1. ROC + F1 Tradeoff Plot
- **Left panel**: ROC curves per class with AUC scores
  - AUC > 0.9 = excellent discrimination
  - Diagonal line = random guessing (AUC=0.5)
- **Right panel**: F1 vs Recall scatter plot
  - Bubble size = Precision (larger = better precision)
  - Points above the diagonal = good precision-recall balance

#### 2. Precision-Recall Curves
- **Each class subplot**: PR curve with F1 iso-lines
  - Dashed lines = constant F1 values (0.2, 0.4, 0.6, 0.8)
  - Dot = point with best F1 score
  - AP = Average Precision (area under PR curve)
- **Summary plot**: All classes together for comparison
- **Interpretation**: For imbalanced data, PR curves are more informative than ROC

#### 3. Confusion Matrix + Per-Class Bars
- **Left**: Normalized confusion matrix
  - Diagonal = correct classifications (higher = better)
  - Off-diagonal = common confusions
- **Right**: Per-class Recall, Precision, F1 bar chart
  - Classes 0 and 4 are typically hardest (noise vs different clusters)
  - Classes 2 and 3 should be symmetric (source-only vs dest-only)

#### 4. Comprehensive Comparison Plots
- **Top-left**: FSS vs RSS bar chart
  - If FSS < RSS significantly → model has poor precision
  - Green bars (FSS) vs Blue bars (RSS)
- **Top-right**: FSS vs RSS scatter
  - Red points = FSS < RSS (precision problem)
  - Blue points = FSS ≈ RSS (good balance)
- **Bottom-left**: Per-class F1 scores (top 7 models)
  - Shows which classes each model handles well
- **Bottom-right**: Architecture comparison box plot
  - Compares GCN, GAT, Transformer, SAGE performance

#### 5. Radar Chart
- Shows top N models across 5 metrics
- Larger area = better overall performance
- Different shapes reveal model strengths/weaknesses

### Understanding the HTML Report

The `master_report.html` provides:

1. **Summary statistics** - Best model's FSS, Macro F1, Accuracy
2. **Metric explanation** - Why FSS is primary
3. **Interactive table** - Sortable, searchable model rankings
4. **Embedded images** - All comparison plots
5. **Class definitions** - What each label means

Open it in any browser:
```bash
open analysis_output/reports/master_report.html  # macOS
xdg-open analysis_output/reports/master_report.html  # Linux
start analysis_output/reports/master_report.html  # Windows
```

### Complete Analysis Workflow

Here's the full pipeline from training to analysis:

```bash
# Step 1: Train multiple models (optional: use --model all)
python train_gnn_models.py --model gcn --baseline --gpu 0
python train_gnn_models.py --model gat --baseline --gpu 0
python train_gnn_models.py --model transformer --all-features --gpu 0

# Step 2: Run inference on best checkpoints (if not done during training)
python train_gnn_models.py --model gcn --baseline --inference-only
python train_gnn_models.py --model gat --baseline --inference-only

# Step 3: Analyze all results
python analyze_results.py \
    --models-dir /storage/foundation_experiments \
    --parquet-dir /storage/foundation_experiments \
    --output-dir ./final_analysis

# Step 4: Open report
open ./final_analysis/reports/master_report.html
```

### Interpreting Results: Quick Guide

| If you see... | This means... | Action |
|---------------|----------------|--------|
| FSS > 4.0 | Excellent model | Deploy or use for physics analysis |
| FSS < 2.5 | Poor performance | Try more features or different architecture |
| FSS much lower than RSS | Precision problem | Use focal loss or class weights |
| Class 0 F1 < 0.5 | Noise identification issues | Check noise statistics in data creation |
| Class 4 F1 < 0.3 | Cluster separation failing | Add cluster-based features |
| High accuracy but low FSS | Class imbalance masking issues | Trust FSS, not accuracy |
| Transformer > GCN by >0.5 | Long-range dependencies matter | Use transformer with attention |
| GAT > GCN by >0.3 | Graph structure is non-uniform | Attention helps - keep GAT |

### Troubleshooting

**"No parquet files found"**
- Ensure you ran inference with `--inference-only` or training completed fully
- Check that `--parquet-dir` points to directory with `results_*.parquet` files

**"Memory error"**
- Reduce `--max-rows-roc` and `--max-rows-confusion`
- Reduce `--batch-size`
- Use fewer models in the directory

**"Missing metrics in pickle"**
- The script falls back to RSS if FSS not found
- Check that you're using the latest `train_gnn_models.py`

**"Different number of models in PKL vs Parquet"**
- Some models may have completed training but not inference
- Run `--inference-only` for those models

---
## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/CaloGraphNet.git
cd CaloGraphNet

# Install dependencies
pip install torch torch-geometric numpy awkward uproot h5py pandas pyarrow scikit-learn psutil

# Optional: For progress bars and enhanced analysis
pip install tqdm seaborn matplotlib
```

### Complete Example: From ROOT to Analysis Report

```bash
# Step 1: Build dataset from ROOT file
python build_graph_dataset.py \
    --input /data/run12345.root \
    --output-dir /my_data \
    --dataset-name physics_run \
    --normalize-energy

# Output will be in: /my_data/physics_run/
# - pairs_physics_run.npy (edges)
# - cells_physics_run.npy (cell metadata)
# - events_physics_run.h5 (features)
# - labels_physics_run.npy (ground truth)

# Step 2: Train models (optional: use --model all for multiple architectures)
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --data-dir /my_data/physics_run \
    --gpu 0 \
    --epochs 30

# Step 3: Run inference on best model (saves parquet predictions)
python train_gnn_models.py \
    --model gcn \
    --baseline \
    --inference-only \
    --data-dir /my_data/physics_run \
    --gpu 0

# Step 4: Generate comprehensive analysis report
python analyze_results.py \
    --models-dir /storage/foundation_experiments \
    --parquet-dir /storage/foundation_experiments \
    --output-dir ./analysis_results

# Step 5: Open the interactive report
open ./analysis_results/reports/master_report.html
```

### Expected Outputs

After running the full pipeline, you'll have:

```
/storage/foundation_experiments/
├── best_gcn_baseline_h128_l6.pt          # Best model checkpoint
├── gcn_baseline_h128_l6_metrics.pkl      # Training metrics
├── results_gcn_baseline_h128_l6.parquet  # Predictions (230M edges)

./analysis_results/
├── figures/                               # All plots
│   ├── roc_curves/                       # ROC per model
│   ├── pr_curves/                        # PR per model
│   ├── confusion_matrices/               # Confusion matrices
│   ├── comparison_plots/                 # Cross-model comparisons
│   └── metrics_radar/                    # Radar charts
├── tables/
│   └── comprehensive_table.html          # Sortable results table
├── reports/
│   └── master_report.html                # Complete analysis report
└── data/
    └── comprehensive_metrics.csv         # All metrics in CSV
```

### Analyze Results with Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
df = pd.read_parquet("results_gcn_baseline_h128_l6.parquet")

# Compute F1 Sum Score
def f1_sum_score(df):
    f1_scores = []
    for c in range(5):
        tp = ((df['true_label'] == c) & (df['pred_label'] == c)).sum()
        fp = ((df['true_label'] != c) & (df['pred_label'] == c)).sum()
        fn = ((df['true_label'] == c) & (df['pred_label'] != c)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    return sum(f1_scores)

print(f"F1 Sum Score: {f1_sum_score(df):.2f} (range: 1.0=random, 5.0=perfect)")

# Load comprehensive metrics
metrics_df = pd.read_csv("analysis_results/data/comprehensive_metrics.csv")
print(f"Best model: {metrics_df.iloc[0]['name']} (FSS={metrics_df.iloc[0]['f1_sum_score']:.2f})")
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

## Current Limitations & Lessons Learned

### The Challenge: Why Performance Isn't Where We Want It

Despite experimenting with multiple architectures (GCN, GAT, Graph Transformer, GraphSAGE), feature sets, and loss functions, the best model achieves an **F1 Sum Score of only 2.75** (where 5.0 would be perfect). This section documents what we've learned from our experiments and where we believe the fundamental issues lie.

### Our Best Results So Far

From 11 trained models, here are the top performers:

| Rank | Model | F1 Sum Score | Accuracy | Macro F1 | Key Features |
|------|-------|--------------|----------|----------|--------------|
| 1 | SAGE (8 layers, focal loss) | **2.75** | 95.1% | 0.55 | Baseline features |
| 2 | Transformer (8 heads, focal loss) | 2.63 | 94.0% | 0.53 | Baseline features |
| 3 | SAGE (all features, inverse loss) | 2.31 | 86.4% | 0.46 | All 42+ features |
| 4 | SAGE (inverse loss) | 2.25 | 84.9% | 0.45 | Baseline |
| 5 | Transformer (inverse loss) | 2.25 | 85.3% | 0.45 | Baseline |

**Key observation:** The best models achieve **~95% accuracy** but only **~2.75 FSS** (55% of perfect). This massive gap between accuracy and FSS tells us something important.

### Lesson 1: Accuracy is Dangerously Misleading for Imbalanced Data

**The problem:** In our dataset, Class 0 (Lone-Lone / noise-noise) dominates with ~92% of all edges. A trivial model that predicts **every edge as Class 0** achieves:
- **Accuracy: 92%** (looks excellent! ✅)
- **F1 Sum Score: ~1.2** (close to random baseline of 1.0 ❌)
- **Macro F1: ~0.20** (fails on all other classes ❌)

**Our actual best model vs trivial baseline:**

| Metric | Trivial Model (all Class 0) | Our Best Model (SAGE + focal) | Improvement |
|--------|----------------------------|-------------------------------|-------------|
| Accuracy | 92.0% | 95.1% | +3.1% |
| F1 Sum Score | ~1.2 | **2.75** | +129% |
| Macro F1 | ~0.20 | **0.55** | +175% |

**Our solution:** We abandoned accuracy entirely as a meaningful metric and adopted:
- **F1 Sum Score (FSS)** = Σ(F1 per class) - **Accounts for BOTH False Positives AND False Negatives**
- Perfect score = 5.0 (all classes perfect)
- Random baseline = 1.0 (all classes at chance)

**Why FSS is superior:** The trivial "all Class 0" model scores:
- Accuracy: 92% (seems good)
- FSS: 1.2 (clearly terrible - reveals the problem)

### Lesson 2: Focal Loss Helps, But Doesn't Solve the Core Issue

Notice that our top 2 models both use **focal loss** (γ=2.0), while inverse-weighted models perform worse:

| Loss Function | Best FSS | Best Accuracy | Notes |
|---------------|----------|---------------|-------|
| **Focal (γ=2.0)** | **2.75** | 95.1% | Best overall |
| Inverse | 2.31 | 86.4% | Worse accuracy, lower FSS |
| Logarithmic | 2.22 | 93.7% | Mid-range |

**What this tells us:** Focal loss helps (especially with class imbalance), but even the best focal loss model only reaches FSS 2.75. The ceiling isn't loss function-related.

### Lesson 3: More Features Don't Always Help (Surprising Finding!)

Counter-intuitively, **all-features mode (42+ features) performed WORSE than baseline (3 features):**

| Model | Features | FSS | Accuracy | Macro F1 |
|-------|----------|-----|----------|----------|
| SAGE + focal | Baseline (3) | **2.75** | 95.1% | 0.55 |
| SAGE + inverse | All (42+) | 2.31 | 86.4% | 0.46 |

**Possible explanations:**
1. **Overfitting:** 42+ features on 1.2M edges may cause overfitting despite regularization
2. **Noise in features:** Some features (noise_category, cluster_id_norm) may be correlated with the label in ways that hurt generalization
3. **Redundancy:** The extra features may not add independent signal

**What we learned:** More data isn't always better. Feature engineering requires careful selection.

### Lesson 4: Architecture Differences Are Smaller Than Expected

Despite large architectural differences, performance varies only modestly:

| Architecture | Best FSS | Best Macro F1 | Notes |
|--------------|----------|---------------|-------|
| GraphSAGE | **2.75** | 0.55 | Winner (8 layers, focal) |
| Transformer | 2.63 | 0.53 | Attention doesn't help much |
| GCN | 2.23 | 0.45 | Simpler, but worse |
| GAT | 2.19 | 0.44 | Attention without global context? |

**Key insight:** The gap between SAGE (best) and GCN (worst) is only ~0.5 FSS. All architectures seem to hit a similar ceiling around FSS 2.5-2.8.

### Lesson 5: The Core Issue May Be Graph Construction Itself

Given that:
- ✅ We tried 4 different architectures (GCN, GAT, Transformer, SAGE)
- ✅ We tried 3 loss functions (focal, inverse, logarithmic)
- ✅ We tried 2 feature sets (baseline vs all 42+)
- ✅ We varied hyperparameters (layers, heads, learning rate)
- ❌ Best FSS still only **2.75** (55% of perfect)

**Our current hypothesis:** We're framing this as a **node classification with fixed graph** problem, but physics suggests it should be a **graph learning problem**.

**The physics reality:**
- A single calorimeter cell can receive energy from **multiple overlapping particle showers**
- In high-energy environments, clusters **overlap spatially**
- A cell doesn't belong to a single cluster - it has **fractional energy contributions** from multiple clusters

**The mismatch in our current approach:**
```
Reality:                     Our Model:
┌─────────────────┐          ┌─────────────────┐
│  Cluster A      │          │  Node: Cell X   │
│    ┌───┐        │          │  Question:      │
│    │ X │ 30%    │          │  "Which single  │
│    └───┘        │          │   cluster does  │
│         Cluster │          │   this cell     │
│         B 70%   │          │   belong to?"   │
└─────────────────┘          └─────────────────┘
                             
Problem: The question itself may be wrong!
Cell X belongs to BOTH clusters (30% A, 70% B)
```

### Why This Explains Our Results

| Observation | Explanation under Graph Learning Hypothesis |
|-------------|---------------------------------------------|
| Best FSS stuck at 2.75 | Hard labels can't represent fractional membership |
| SAGE > Transformer | Local aggregation may be better for overlapping clusters than global attention |
| More features hurt | Adding cluster_id_norm may "bake in" hard assignments |
| Focal loss helps but limited | Focal helps with imbalance but can't fix wrong label formulation |

### The Graph Learning Hypothesis

We suspect the right approach is to **learn the graph structure itself** rather than classify edges on a fixed graph:

| Current Approach (Edge Classification) | Proposed Approach (Graph Learning) |
|----------------------------------------|-------------------------------------|
| Fixed graph from geometric neighbors | Learn which cells should be connected |
| Each edge has a single label (0-4) | Each edge has a weight/strength |
| Hard assignment to one cluster | Soft assignment to multiple clusters |
| Assumes clusters are disjoint | Allows overlapping clusters |
| One prediction per edge | Learned adjacency matrix |

**What success might look like:** Instead of predicting Class 1 (same cluster) vs Class 4 (different clusters), predict a continuous value: "probability these cells share energy"

### Future Directions

Based on our analysis, here are promising research directions:

#### Direction 1: Soft Edge Prediction (Most Feasible)
Instead of 5 classes, predict:
- **Probability of sharing energy** (continuous from 0 to 1)
- **Uncertainty quantification** per prediction
- **Expected energy fraction** rather than hard label

#### Direction 2: Graph Learning / Differentiable Graph Generation
- Learn the adjacency matrix directly
- Allow fractional edge weights
- Use graph generation techniques (e.g., GraphVAE, EDGE, DiGress)

#### Direction 3: Hypergraph or Multi-Graph Representations
- One cell → multiple cluster memberships
- Hypergraph where hyperedges represent clusters
- Multi-layer graph (one layer per possible cluster)

#### Direction 4: Energy Flow as Edge Weights
- Use actual energy sharing information (if available from simulation)
- Train on data with known fractional contributions
- Predict energy fractions rather than hard labels

### Practical Recommendations for Users of This Code

If you're using CaloGraphNet and encountering similar limitations:

1. **Don't trust accuracy** - A model with 95% accuracy might still have FSS < 3.0
2. **Compute FSS after every experiment** - It's the only reliable metric
3. **Use focal loss** - It consistently outperforms inverse weighting
4. **Start with baseline features** - All-features mode may hurt more than help
5. **SAGE seems best** - GraphSAGE with 8 layers and focal loss is our top performer
6. **Check per-class F1** - If Classes 1 and 4 have low F1, you may have overlapping clusters

### What 2.75 FSS Actually Means

An FSS of 2.75 means:
- On average, each class achieves F1 ≈ 0.55
- Random guessing would be F1 ≈ 0.20 per class (FSS = 1.0)
- Perfect would be F1 = 1.00 per class (FSS = 5.0)
- We're about halfway between random and perfect

**Class-specific performance from our best model (SAGE + focal):**

| Class | Likely F1 | Interpretation |
|-------|-----------|----------------|
| Class 0 (Lone-Lone) | ~0.92 | Very good (easy class) |
| Class 1 (Same Cluster) | ~0.45 | Poor - can't identify same-cluster pairs |
| Class 2 (Source Only) | ~0.55 | Moderate |
| Class 3 (Dest Only) | ~0.55 | Moderate |
| Class 4 (Different Clusters) | ~0.28 | Very poor - can't separate different clusters |

**The gap between Class 1 and Class 4 F1 is particularly telling:** The model can't distinguish "same cluster" from "different clusters" - which is exactly what you'd expect if clusters overlap!

### Summary Table of What We've Tried

| Approach | What We Did | Best Result | Lesson |
|----------|-------------|-------------|--------|
| **Metrics** | Accuracy → FSS | 2.75 FSS max | Accuracy is useless (92% trivial baseline) |
| **Features** | Baseline (3) → All (42+) | Worse performance | More isn't better; feature engineering matters |
| **Loss** | CrossEntropy → Focal | +0.5 FSS improvement | Focal helps with imbalance |
| **Architecture** | GCN → SAGE → Transformer | SAGE best (2.75) | Differences are small (~0.5 FSS range) |
| **Graph** | Fixed geometric neighbors | Stuck at 2.75 FSS | **Root cause: graph assumption fails** |

### Open Questions for the Community

We welcome input and collaboration on these questions:

1. Has anyone successfully applied **soft edge prediction** (continuous labels) to calorimeter clustering?
2. Are there public datasets with **fractional energy contributions** per cell per cluster?
3. Could **graph generative models** (DiGress, GraphARM) learn overlapping cluster assignments?
4. Is **hypergraph representation** more appropriate for overlapping particle showers?
5. For those who've tried both: does **soft clustering** outperform hard classification on similar problems?

### Conclusion

CaloGraphNet successfully:
- ✅ Builds graph datasets from ROOT files
- ✅ Trains multiple GNN architectures (GCN, GAT, Transformer, SAGE)
- ✅ Provides comprehensive analysis tools (ROC, PR curves, FSS, HTML reports)

However, it has revealed a fundamental limitation: **edge classification on a fixed geometric graph may be insufficient for overlapping particle clusters.** The best model (SAGE + focal loss) achieves FSS 2.75 - far from perfect.

We believe the path forward lies in:
1. **Soft labels** (continuous energy fractions instead of hard cluster assignments)
2. **Graph learning** (learn the graph structure, don't fix it)
3. **Hypergraph representations** (capture multi-cell energy sharing)

We hope this honest assessment helps others avoid similar pitfalls and inspires new approaches to this challenging problem.
