#!/usr/bin/env python3
"""
calo_analysis_suite.py - Unified Analysis & Visualization Suite

Combines:
1. Model performance metrics (ROC, PR, confusion matrices, LOSS CURVES)
2. Event visualization with true detector geometry
3. Masking visualization (pretraining diagnostics) - FULLY CORRECTED
4. Cluster reconstruction validation (union-find + Hungarian matching)
5. Integrated HTML report with links to PDFs

IMPROVEMENTS ADDED:
- Cut on 1-P(Lone-Lone) instead of P(True-True) for clustering
- Truth-based cluster recovery metrics
- IoU distribution plot for optimal threshold selection
- Validation of clustering algorithm on truth pairs
- Efficiency vs η/φ spatial plots (NOW FULLY WIRED)
- Energy-weighted purity/efficiency metrics (NOW FULLY WIRED)
- Debug mode processes ALL phases with 1 event only

Usage:
    python final_results_analysis.py \
        --models-dir ./pkl_files \
        --parquet-dir ./parquet_files \
        --data-dir . \
        --output-dir ./analysis_output_3k \
        --h5-file events_3k_test.h5 \
        --event-offset 2900 \
        --model-type sage \
        --hidden-dim 128 \
        --num-layers 8
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import argparse
import pickle
import glob
import gc
import json
import warnings
import traceback
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
from math import pi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import pyarrow.parquet as pq
import h5py
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, TwoSlopeNorm
from matplotlib.patches import Patch, Circle, Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
from torch.nn import BatchNorm1d
from torch_geometric.utils import to_undirected

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
CLASS_NAMES = {
    0: 'Lone-Lone (Noise)',
    1: 'True-True (Same Cluster)',
    2: 'Cluster-Lone (Source Only)',
    3: 'Lone-Cluster (Dest Only)',
    4: 'Cluster-Cluster (Different)'
}
CLASS_COLORS = ['#2c3e50', '#27ae60', '#e74c3c', '#e67e22', '#9b59b6']
CLASS_CMAP = ListedColormap(CLASS_COLORS)

# Detector geometry
ETA_RANGE = (-3.5, 3.5)
PHI_RANGE = (-3.5, 3.5)
EPS = 1e-6
ENERGY_THRESHOLD = 1e-6
ETA_WIDTH = 0.025
PHI_WIDTH = 0.1

# Priority labeling
USE_PRIORITY_LABELING = True
PRIORITY_LOOKUP = np.zeros(5, dtype=int)
PRIORITY_LOOKUP[1] = 5
PRIORITY_LOOKUP[4] = 4
PRIORITY_LOOKUP[3] = 3
PRIORITY_LOOKUP[2] = 3
PRIORITY_LOOKUP[0] = 1
PRIORITY_ORDER = [1, 4, 3, 2, 0]

TILE_HEC_LAYERS = [8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23]

# Cell sizes dictionary
CELL_SIZES = {
    '0_0': (0.025, 0.1),
    '0_1': (0.0031, 0.0245),
    '0_2': (0.025, 0.0245),
    '0_3': (0.05, 0.0245),
    '0_4': (0.025, 0.1),
    '0_5': (0.0031, 0.1),
    '0_6': (0.1, 0.1),
    '0_7': (0.1, 0.1),
    '1_8': (0.1, 0.09817481),
    '1_9': (0.1, 0.09817481),
    '1_10': (0.1, 0.09817481),
    '1_11': (0.1, 0.09817481),
    '2_21': (0.1, 0.1),
    '2_22': (0.1, 0.1),
    '2_23': (0.1, 0.1),
    '3_12': (0.1, 0.09817481),
    '3_13': (0.1, 0.09817481),
    '3_14': (0.2, 0.09817481),
    '3_15': (0.2, 0.09817481),
    '3_16': (0.2, 0.09817481),
    '3_17': (0.25, 0.09817481),
    '3_18': (0.1, 0.09817481),
    '3_19': (0.1, 0.09817481),
    '3_20': (0.2, 0.09817481),
}

# Layer groups
LAYER_GROUPS = [
    {"name": "Complete System", "layers": "all"},
    {"name": "EM Presampler", "layers": [(0, 0), (0, 4)]},
    {"name": "EM Layer 1", "layers": [(0, 1), (0, 5)]},
    {"name": "EM Layer 2", "layers": [(0, 2), (0, 6)]},
    {"name": "EM Layer 3", "layers": [(0, 3), (0, 7)]},
    {"name": "Tile A", "layers": [(1, 8), (3, 12), (3, 13), (3, 18)]},
    {"name": "Tile BC", "layers": [(1, 9), (1, 10), (3, 14), (3, 15), (3, 19)]},
    {"name": "Tile D+HEC", "layers": [(1, 11), (3, 16), (3, 17), (3, 20), (2, 21), (2, 22), (2, 23)]},
]

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
GREEN_COLORMAP = LinearSegmentedColormap.from_list("green_grad", ["#e8f5e9", "#1b5e20"])
BLUE_COLORMAP = LinearSegmentedColormap.from_list("blue_grad", ["#e3f2fd", "#0d47a1"])


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_cell_size(subCalo, layer):
    """Get true cell dimensions for a given layer."""
    key = f"{subCalo}_{layer}"
    return CELL_SIZES.get(key, (0.1, 0.1))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified CaloGraph Analysis & Visualization Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--models-dir', type=str, required=True,
                        help='Directory containing model pickle files (*_metrics.pkl)')
    parser.add_argument('--parquet-dir', type=str, required=True,
                        help='Directory containing parquet prediction files')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing HDF5 event data')
    parser.add_argument('--output-dir', type=str, default='./analysis_output',
                        help='Output directory for results')
    parser.add_argument('--debug', action='store_true',
                    help='Debug mode: process ALL phases with only 1 event')
    parser.add_argument('--max-models-viz', type=int, default=None,
                    help='Maximum number of models to visualize (default: all)')
    parser.add_argument('--max-rows-roc', type=int, default=5_000_000,
                        help='Max rows for ROC/PR computation (default: 5M = ~4 events)')
    parser.add_argument('--max-rows-confusion', type=int, default=10_000_000,
                        help='Max rows for confusion matrix (default: 10M = ~8 events)')
    parser.add_argument('--max-events-cluster', type=int, default=None,
                    help='Max events for cluster validation (default: all events)')
    parser.add_argument('--batch-size', type=int, default=100000)
    parser.add_argument('--cluster-thresholds', type=str, default='0.5,0.7,0.9')
    parser.add_argument('--events-to-visualize', type=str, 
                        default='typical:5,worst:5,best:3,high_multiplicity:2')
    parser.add_argument('--include-masking-viz', action='store_true',
                        help='Include masking visualization panels')
    parser.add_argument('--include-cc-ablation', action='store_true',
                        help='Run CC-confusion ablation study (Options A and B) for cluster reconstruction')
    parser.add_argument('--cc-heavy-threshold', type=float, default=0.5,
                        help='Fraction of CC-true edges for a cell to be considered "CC-heavy" (default: 0.5)')
    parser.add_argument('--auto-masking-viz', action='store_true',
                        help='Auto-detect pretrained checkpoints per model (by filename)')
    parser.add_argument('--delta-r', type=float, default=0.05,
                    help='Maximum ΔR for cluster matching (default: 0.05)')
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Path to pretrained model checkpoint for masking visualization')
    parser.add_argument('--mask-type', type=str, default='random',
                        choices=['random', 'feature', 'geometry', 'cluster'])
    parser.add_argument('--mask-ratio', type=float, default=0.15)
    parser.add_argument('--h5-file', type=str, default='events.h5',
                        help='HDF5 filename in data-dir (default: events.h5)')
    parser.add_argument('--model-type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'transformer', 'sage'])
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=2)
    parser.add_argument('--visualization-threshold', type=float, default=0.7,
                        help='Confidence threshold for cluster visualization (default: 0.7)')
    parser.add_argument('--event-offset', type=int, default=0,
                        help='Offset to subtract from parquet event IDs to get HDF5 indices')
    parser.add_argument('--use-not-noise-cut', action='store_true',
                        help='Use 1-P(Lone-Lone) cut instead of P(True-True) for clustering')
    
    return parser.parse_args()


def create_output_structure(output_dir):
    """Create organized folder structure."""
    folders = {
        'roc_curves': os.path.join(output_dir, 'figures', 'roc_curves'),
        'pr_curves': os.path.join(output_dir, 'figures', 'pr_curves'),
        'confusion_matrices': os.path.join(output_dir, 'figures', 'confusion_matrices'),
        'comparison_plots': os.path.join(output_dir, 'figures', 'comparison_plots'),
        'loss_curves': os.path.join(output_dir, 'figures', 'loss_curves'),
        'event_pdfs': os.path.join(output_dir, 'event_pdfs'),
        'cluster_validation': os.path.join(output_dir, 'cluster_validation'),
        'masking_viz': os.path.join(output_dir, 'figures', 'masking_viz'),
        'tables': os.path.join(output_dir, 'tables'),
        'reports': os.path.join(output_dir, 'reports'),
        'data': os.path.join(output_dir, 'data'),
        'binary_metrics': os.path.join(output_dir, 'figures', 'binary_metrics'),
        'cluster_ablation': os.path.join(output_dir, 'cluster_validation', 'ablation'),
        'iou_distributions': os.path.join(output_dir, 'cluster_validation', 'iou_distributions'),
        'efficiency_maps': os.path.join(output_dir, 'cluster_validation', 'efficiency_maps'),
        'energy_metrics': os.path.join(output_dir, 'cluster_validation', 'energy_metrics'),
        'delta_r_matching': os.path.join(output_dir, 'cluster_validation', 'delta_r_matching'),
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    return folders


def discover_pretrained_checkpoints(models_dir):
    """Scan models_dir for pretrained_*.pt checkpoints."""
    discovered = {}
    for ckpt_path in glob.glob(os.path.join(models_dir, "pretrained_*.pt")):
        fname = os.path.basename(ckpt_path)
        model_name = fname[len("pretrained_"):-len(".pt")]
        mask_type = None
        for candidate in ['random', 'feature', 'geometry', 'cluster']:
            if f"_{candidate}_" in fname or fname.endswith(f"_{candidate}"):
                mask_type = candidate
                break
        discovered[model_name] = {'checkpoint': ckpt_path, 'mask_type': mask_type}
    return discovered


# ============================================================================
# MASKING INFERENCE ENGINE
# ============================================================================
class MaskingInferenceEngine:
    """Runs masked inference using a pretrained model for visualization."""
    
    def __init__(self, model_path, model_type='gcn', hidden_dim=128,
                 num_layers=6, num_heads=2, device='cuda'):
        self.model_path = model_path
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_names = None
        self.input_dim = None
        
        self._load_model()
    
    def _load_model(self):
        """Load pretrained model checkpoint."""
        if not os.path.exists(self.model_path):
            print(f"  ⚠️ Pretrained model not found: {self.model_path}")
            return
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        
        self.input_dim = checkpoint.get('input_dim', 7)
        self.feature_names = checkpoint.get(
            'feature_names',
            ['snr_scaled', 'snr_gt4', 'snr_gt2', 'snr_gt0', 'eta', 'sin_phi', 'cos_phi']
        )
        
        class GraphEncoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, model_type, num_heads):
                super().__init__()
                self.model_type = model_type
                self.node_embedding = nn.Linear(input_dim, hidden_dim)
                self.convs = nn.ModuleList()
                
                for _ in range(num_layers):
                    if model_type == 'gcn':
                        self.convs.append(GCNConv(hidden_dim, hidden_dim))
                    elif model_type == 'gat':
                        self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads,
                                                 heads=num_heads, edge_dim=5))
                    elif model_type == 'sage':
                        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                    elif model_type == 'transformer':
                        self.convs.append(TransformerConv(hidden_dim, hidden_dim // num_heads,
                                                         heads=num_heads, edge_dim=5))
                
                self.bns = nn.ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers)])
                self.reconstruction_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, input_dim)
                )
            
            def forward(self, x, edge_index, edge_attr=None):
                x = self.node_embedding(x)
                for conv, bn in zip(self.convs, self.bns):
                    if edge_attr is not None and self.model_type in ['gat', 'transformer']:
                        h = torch.relu(bn(conv(x, edge_index, edge_attr)))
                    else:
                        h = torch.relu(bn(conv(x, edge_index)))
                    x = x + h
                return self.reconstruction_head(x)
        
        self.model = GraphEncoder(
            self.input_dim, self.hidden_dim, self.num_layers,
            self.model_type, self.num_heads
        ).to(self.device)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        model_dict = self.model.state_dict()
        filtered_dict = {}
        skipped = []
        for k, v in state_dict.items():
            k_clean = k.replace('module.', '')
            if 'fc.' in k_clean:
                continue
            if k_clean in model_dict:
                if v.shape == model_dict[k_clean].shape:
                    filtered_dict[k_clean] = v
                else:
                    skipped.append((k_clean, tuple(v.shape), tuple(model_dict[k_clean].shape)))
        
        self.model.load_state_dict(filtered_dict, strict=False)
        self.model.eval()
        print(f"  ✓ Loaded pretrained model (transferred {len(filtered_dict)} parameters)")
        if skipped:
            print(f"  ⚠️ {len(skipped)} parameter(s) skipped due to shape mismatch")
    
    def generate_masking_data(self, df_cells, edges_df, event_id, h5_path,
                               mask_type='random', mask_ratio=0.15, event_offset=0):
        """Generate masking and reconstruction data for an event."""
        if self.model is None:
            return None, None
        
        h5_event_id = event_id - event_offset
        
        has_snr_scaled = False
        snr_row = None
        
        with h5py.File(h5_path, 'r') as h5f:
            if 'cell/cell_SNR_scaled' in h5f:
                snr_row = h5f['cell/cell_SNR_scaled'][h5_event_id]
                has_snr_scaled = True
            elif 'cell/snr_computed' in h5f:
                snr_row = h5f['cell/snr_computed'][h5_event_id]
            elif 'cell/cell_SNR_raw' in h5f:
                snr_row = h5f['cell/cell_SNR_raw'][h5_event_id]
            elif 'cell/snr_raw' in h5f:
                snr_row = h5f['cell/snr_raw'][h5_event_id]
            elif 'cell/cell_e' in h5f:
                snr_row = h5f['cell/cell_e'][h5_event_id]
            elif 'cell/energy_raw' in h5f:
                snr_row = h5f['cell/energy_raw'][h5_event_id]
            else:
                snr_row = np.zeros(len(df_cells), dtype=np.float32)
        
        original_positions = df_cells.index.to_numpy()
        snr = snr_row[original_positions].astype(np.float32)
        
        eta = df_cells['eta'].values
        phi = df_cells['phi'].values
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        
        if has_snr_scaled:
            snr_scaled = snr
        else:
            snr_scaled = np.sign(snr) * np.log1p(np.abs(snr))
        
        features = np.stack([
            snr_scaled,
            (np.abs(snr) > 4).astype(np.float32),
            (np.abs(snr) > 2).astype(np.float32),
            (np.abs(snr) > 0).astype(np.float32),
            eta, sin_phi, cos_phi
        ], axis=1).astype(np.float32)
        
        x = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        pos_to_row = {pos: row for row, pos in enumerate(original_positions)}
        
        if edges_df is not None and not edges_df.empty:
            valid = edges_df['source_id'].isin(pos_to_row) & edges_df['target_id'].isin(pos_to_row)
            edges_valid = edges_df[valid]
            src_rows = edges_valid['source_id'].map(pos_to_row).values
            dst_rows = edges_valid['target_id'].map(pos_to_row).values
            edge_index_np = np.stack([src_rows, dst_rows])
            edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(self.device)
            edge_index = to_undirected(edge_index)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long).to(self.device)
        
        mask_full = self._create_mask(len(df_cells), mask_type, mask_ratio, df_cells)
        mask_tensor = torch.tensor(mask_full, dtype=torch.bool).to(self.device)
        
        x_masked = x.clone()
        x_masked[mask_tensor] = 0.0

        with torch.no_grad():
            reconstructed = self.model(x_masked, edge_index)

        orig_np = x.cpu().numpy()
        masked_input_np = x_masked.cpu().numpy()
        recon_np = reconstructed.cpu().numpy()

        errors_full = np.zeros(len(df_cells))
        if mask_full.any():
            errors_full[mask_full] = recon_np[mask_full, 0] - orig_np[mask_full, 0]

        mask_info = {
            'is_masked': pd.Series(mask_full, index=original_positions)
        }
        recon_info = {
            'errors': pd.Series(errors_full, index=original_positions),
            'is_masked': pd.Series(mask_full, index=original_positions),
            'input_values': pd.Series(masked_input_np[:, 0], index=original_positions),
            'true_values': pd.Series(orig_np[:, 0], index=original_positions),
            'recon_values': pd.Series(recon_np[:, 0], index=original_positions),
        }

        return mask_info, recon_info
    
    def _create_mask(self, n_cells, mask_type, mask_ratio, df_cells):
        """Create masking pattern."""
        rng = np.random.RandomState(42)
        
        if mask_type == 'random':
            return rng.random(n_cells) < mask_ratio
        elif mask_type == 'geometry':
            seed_idx = rng.randint(0, n_cells)
            seed_eta = df_cells.iloc[seed_idx]['eta']
            seed_phi = df_cells.iloc[seed_idx]['phi']
            deta = np.abs(df_cells['eta'].values - seed_eta)
            dphi = np.abs(df_cells['phi'].values - seed_phi)
            dphi = np.minimum(dphi, 2 * np.pi - dphi)
            in_window = (deta < 0.3) & (dphi < 0.3)
            return in_window & (rng.random(n_cells) < mask_ratio * 2)
        elif mask_type == 'cluster':
            cluster_indices = df_cells['cluster_index'].values
            unique_clusters = np.unique(cluster_indices[cluster_indices > 0])
            if len(unique_clusters) > 0:
                n_mask = max(1, int(len(unique_clusters) * mask_ratio))
                clusters_to_mask = rng.choice(unique_clusters, n_mask, replace=False)
                return np.isin(cluster_indices, clusters_to_mask)
            else:
                return rng.random(n_cells) < mask_ratio
        else:
            return rng.random(n_cells) < mask_ratio


# ============================================================================
# COMPONENT 1: ModelMetricsAnalyzer
# ============================================================================
class ModelMetricsAnalyzer:
    """Handles ROC/PR curves, confusion matrices, loss curves."""

    def __init__(self, models_dir, parquet_dir, output_dir, 
                 max_rows_roc=5_000_000, max_rows_confusion=10_000_000, batch_size=100000):
        self.models_dir = models_dir
        self.parquet_dir = parquet_dir
        self.output_dir = output_dir
        self.max_rows_roc = max_rows_roc
        self.max_rows_confusion = max_rows_confusion
        self.batch_size = batch_size
        self.folders = create_output_structure(output_dir)
    
    def extract_model_metrics(self, metrics):
        """Extract comprehensive metrics from training output."""
        extracted = {
            'f1_sum_score': metrics.get('best_f1_sum_score', 0.0),
            'weighted_f1_score': metrics.get('best_weighted_f1_score', 0.0),
            'macro_f1': metrics.get('best_macro_f1', 0.0),
            'weighted_f1': metrics.get('best_weighted_f1', 0.0),
            'randomness_metric': metrics.get('best_randomness_metric', 0.0),
            'macro_recall': metrics.get('best_macro_recall', 0.0),
            'weighted_recall': metrics.get('best_weighted_recall', 0.0),
            'macro_precision': metrics.get('best_macro_precision', 0.0),
            'weighted_precision': metrics.get('best_weighted_precision', 0.0),
            'accuracy': metrics.get('best_accuracy', 0.0),
            'train_losses': metrics.get('train_loss', []),
            'val_losses': metrics.get('val_loss', []),
            'test_losses': metrics.get('test_loss', []),
            'test_epochs': metrics.get('test_epochs', []),
            'best_epoch': metrics.get('best_epoch', '?'),
            'total_time': metrics.get('total_time', 0.0),
            'pretrained': False,
            'pretrain_mask_type': None,
            'pretrain_mask_ratio': None,
            'per_class_recall': {},
            'per_class_precision': {},
            'per_class_f1': {},
        }
        
        for c in range(5):
            for key in [f'best_recall_class_{c}', f'test_recall_class_{c}']:
                if key in metrics:
                    val = metrics[key]
                    extracted['per_class_recall'][c] = float(max(val)) if isinstance(val, list) and val else float(val)
                    break
            else:
                extracted['per_class_recall'][c] = 0.0
            
            for key in [f'best_precision_class_{c}', f'test_precision_class_{c}']:
                if key in metrics:
                    val = metrics[key]
                    extracted['per_class_precision'][c] = float(max(val)) if isinstance(val, list) and val else float(val)
                    break
            else:
                extracted['per_class_precision'][c] = 0.0
            
            for key in [f'best_f1_class_{c}', f'test_f1_class_{c}']:
                if key in metrics:
                    val = metrics[key]
                    extracted['per_class_f1'][c] = float(max(val)) if isinstance(val, list) and val else float(val)
                    break
            else:
                extracted['per_class_f1'][c] = 0.0
        
        model_args = metrics.get('args', {})
        if model_args.get('pretrain', False):
            extracted['pretrained'] = True
            extracted['pretrain_mask_type'] = model_args.get('mask_type', 'unknown')
            extracted['pretrain_mask_ratio'] = model_args.get('mask_ratio', 0.0)
        
        return extracted
    
    def compute_roc_data(self, model_name, debug=False, debug_event_id=None):
        """Compute ROC curve data using cross-event sampling."""
        parquet_path = os.path.join(self.parquet_dir, f"results_{model_name}.parquet")
        if not os.path.exists(parquet_path):
            return None, None
        
        parquet_file = pq.ParquetFile(parquet_path)
        y_true_all = []
        y_scores_all = []
        total_rows = 0
        score_cols = ['score_class_0', 'score_class_1', 'score_class_2', 
                      'score_class_3', 'score_class_4']
        
        if debug and debug_event_id is not None:
            # Debug: only read one event
            filters = [("event_id", "=", debug_event_id)]
        else:
            filters = None
        
        event_ids = pq.read_table(parquet_path, columns=['event_id'], 
                                  filters=filters).to_pandas()['event_id'].unique()
        n_events = len(event_ids)
        
        if debug:
            rows_per_event = self.max_rows_roc  # Read everything from this event
        else:
            rows_per_event = min(self.max_rows_roc // max(1, n_events), 100_000)
            if rows_per_event < 10_000:
                rows_per_event = 10_000
        
        print(f"      Sampling ~{rows_per_event:,} rows from each of {n_events} events")
        
        for batch in parquet_file.iter_batches(
            batch_size=self.batch_size,
            columns=['event_id', 'true_label'] + score_cols
        ):
            chunk = batch.to_pandas()
            
            for event_id, event_chunk in chunk.groupby('event_id'):
                if len(event_chunk) > rows_per_event and not debug:
                    event_chunk = event_chunk.sample(rows_per_event, random_state=42)
                y_true_all.append(event_chunk['true_label'].values)
                y_scores_all.append(event_chunk[score_cols].values)
                total_rows += len(event_chunk)
            
            if total_rows >= self.max_rows_roc and not debug:
                break
            elif debug and total_rows > 0:
                break  # Only first batch needed for debug
        
        if not y_true_all:
            return None, None
        
        y_true = np.concatenate(y_true_all)
        y_scores = np.concatenate(y_scores_all)
        return y_true, y_scores
    
    def compute_confusion_matrix(self, model_name, debug=False, debug_event_id=None):
        """Compute confusion matrix."""
        parquet_path = os.path.join(self.parquet_dir, f"results_{model_name}.parquet")
        if not os.path.exists(parquet_path):
            return None
        
        parquet_file = pq.ParquetFile(parquet_path)
        cm = np.zeros((5, 5), dtype=np.int64)
        total_processed = 0
        
        if debug and debug_event_id is not None:
            filters = [("event_id", "=", debug_event_id)]
        else:
            filters = None
        
        event_ids = pq.read_table(parquet_path, columns=['event_id'], 
                                  filters=filters).to_pandas()['event_id'].unique()
        n_events = len(event_ids)
        
        if debug:
            rows_per_event = self.max_rows_confusion
        else:
            rows_per_event = min(self.max_rows_confusion // max(1, n_events), 200_000)
            if rows_per_event < 20_000:
                rows_per_event = 20_000
        
        for batch in parquet_file.iter_batches(
            batch_size=self.batch_size,
            columns=['event_id', 'true_label', 'pred_label']
        ):
            chunk = batch.to_pandas()
            
            for event_id, event_chunk in chunk.groupby('event_id'):
                if len(event_chunk) > rows_per_event and not debug:
                    event_chunk = event_chunk.sample(rows_per_event, random_state=42)
                y_true = event_chunk['true_label'].values
                y_pred = event_chunk['pred_label'].values
                np.add.at(cm, (y_true, y_pred), 1)
                total_processed += len(event_chunk)
            
            if total_processed >= self.max_rows_confusion and not debug:
                break
            elif debug and total_processed > 0:
                break
        
        return cm
    
    def plot_roc_curves(self, model_name, extracted, debug=False, debug_event_id=None):
        """Generate ROC curves."""
        y_true, y_scores = self.compute_roc_data(model_name, debug, debug_event_id)
        if y_true is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        ax = axes[0]
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
        
        for i in range(5):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=CLASS_COLORS[i], lw=2,
                    label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {model_name}')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        f1_scores = [extracted['per_class_f1'].get(c, 0) for c in range(5)]
        recall_scores = [extracted['per_class_recall'].get(c, 0) for c in range(5)]
        precision_scores = [extracted['per_class_precision'].get(c, 0) for c in range(5)]
        sizes = [max(50, 200 * p) for p in precision_scores]
        
        for i in range(5):
            ax2.scatter(recall_scores[i], f1_scores[i], s=sizes[i],
                       color=CLASS_COLORS[i], alpha=0.7, edgecolors='black', linewidth=1.5)
            ax2.annotate(CLASS_NAMES[i].split(' (')[0],
                        (recall_scores[i], f1_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('F1 Score')
        ax2.set_title(f'Precision-Recall Tradeoff (FSS={extracted["f1_sum_score"]:.2f})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['roc_curves'], f"{model_name}_roc_f1.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        del y_true, y_scores, y_true_bin
        gc.collect()

    def plot_rejection_curves(self, model_name, extracted, debug=False, debug_event_id=None):
        """Background-rejection curves."""
        y_true, y_scores = self.compute_roc_data(model_name, debug, debug_event_id)
        if y_true is None:
            return
    
        fig, ax = plt.subplots(figsize=(10, 8))
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    
        for i in range(5):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            nonzero = fpr > 0
            fpr_nz = fpr[nonzero]
            tpr_nz = tpr[nonzero]
            if len(fpr_nz) == 0:
                continue
            rejection = 1.0 / fpr_nz
            ax.plot(tpr_nz, rejection, color=CLASS_COLORS[i], lw=2,
                    label=f'{CLASS_NAMES[i]}')
    
        ax.set_yscale('log')
        ax.set_xlabel('Signal Efficiency (TPR)')
        ax.set_ylabel('Background Rejection (1/FPR)')
        ax.set_title(f'Rejection Curves - {model_name}')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
    
        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['roc_curves'], f"{model_name}_rejection.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        del y_true, y_scores, y_true_bin
        gc.collect()
    
    def plot_pr_curves(self, model_name, extracted, debug=False, debug_event_id=None):
        """Generate Precision-Recall curves."""
        y_true, y_scores = self.compute_roc_data(model_name, debug, debug_event_id)
        if y_true is None:
            return
        
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        all_aps = []
        
        for i in range(5):
            ax = axes[i]
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_scores[:, i])
            all_aps.append(ap)
            ax.plot(recall, precision, color=CLASS_COLORS[i], lw=2, label=f'AP={ap:.3f}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'{CLASS_NAMES[i]} (AP={ap:.3f})')
            ax.legend(loc='lower left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
        
        ax_summary = axes[5]
        for i in range(5):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
            ax_summary.plot(recall, precision, color=CLASS_COLORS[i], lw=2, alpha=0.7,
                           label=f'{CLASS_NAMES[i].split(" (")[0]}')
        ax_summary.set_xlabel('Recall')
        ax_summary.set_ylabel('Precision')
        ax_summary.set_title(f'All Classes (Mean AP: {np.mean(all_aps):.3f})')
        ax_summary.legend(loc='lower left', fontsize=7)
        ax_summary.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['pr_curves'], f"{model_name}_pr_curves.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        del y_true, y_scores, y_true_bin
        gc.collect()
    
    def plot_confusion_matrix(self, model_name, extracted, debug=False, debug_event_id=None):
        """Generate confusion matrix."""
        cm = self.compute_confusion_matrix(model_name, debug, debug_event_id)
        if cm is None:
            return
        
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        ax = axes[0]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=[CLASS_NAMES[i] for i in range(5)],
                    yticklabels=[CLASS_NAMES[i] for i in range(5)],
                    vmin=0, vmax=1, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {model_name}')
        
        ax2 = axes[1]
        x = np.arange(5)
        width = 0.25
        recall_vals = [extracted['per_class_recall'].get(c, 0) for c in range(5)]
        precision_vals = [extracted['per_class_precision'].get(c, 0) for c in range(5)]
        f1_vals = [extracted['per_class_f1'].get(c, 0) for c in range(5)]
        
        ax2.bar(x - width, recall_vals, width, label='Recall', color='#3498db', alpha=0.8)
        ax2.bar(x, precision_vals, width, label='Precision', color='#2ecc71', alpha=0.8)
        ax2.bar(x + width, f1_vals, width, label='F1', color='#e74c3c', alpha=0.8)
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Score')
        ax2.set_title(f'Per-Class Performance (FSS={extracted["f1_sum_score"]:.2f})')
        ax2.set_xticks(x)
        ax2.set_xticklabels([CLASS_NAMES[i].split(' (')[0] for i in range(5)], 
                            rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['confusion_matrices'], 
                                f"{model_name}_confusion.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()

    @staticmethod
    def binarize_labels(y_true, y_scores, positive_class=1):
        """Collapse 5-class labels/scores into binary."""
        y_true_bin = (y_true == positive_class).astype(np.int64)
        y_scores_bin = np.stack([
            1.0 - y_scores[:, positive_class],
            y_scores[:, positive_class],
        ], axis=1)
        return y_true_bin, y_scores_bin

    def plot_roc_curves_binary(self, model_name, debug=False, debug_event_id=None):
        """Binary ROC: class 1 vs. all other classes combined."""
        y_true, y_scores = self.compute_roc_data(model_name, debug, debug_event_id)
        if y_true is None:
            return

        y_true_bin, y_scores_bin = self.binarize_labels(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(8, 8))
        fpr, tpr, _ = roc_curve(y_true_bin, y_scores_bin[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CLASS_COLORS[1], lw=2,
                label=f'Same Cluster vs Rest (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Binary ROC (Same-Cluster vs Rest) - {model_name}')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['binary_metrics'], f"{model_name}_roc_binary.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        del y_true, y_scores, y_true_bin, y_scores_bin
        gc.collect()

    def plot_pr_curves_binary(self, model_name, debug=False, debug_event_id=None):
        """Binary Precision-Recall: class 1 vs. rest."""
        y_true, y_scores = self.compute_roc_data(model_name, debug, debug_event_id)
        if y_true is None:
            return

        y_true_bin, y_scores_bin = self.binarize_labels(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(8, 8))
        precision, recall, _ = precision_recall_curve(y_true_bin, y_scores_bin[:, 1])
        ap = average_precision_score(y_true_bin, y_scores_bin[:, 1])
        ax.plot(recall, precision, color=CLASS_COLORS[1], lw=2, label=f'AP={ap:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Binary PR (Same-Cluster vs Rest) - {model_name}')
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['binary_metrics'], f"{model_name}_pr_binary.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        del y_true, y_scores, y_true_bin, y_scores_bin
        gc.collect()

    @staticmethod
    def _collapse_confusion_matrix_binary(cm, positive_class=1):
        """Collapse a 5x5 confusion matrix into 2x2."""
        n = cm.shape[0]
        other_idx = [i for i in range(n) if i != positive_class]
        cm2 = np.zeros((2, 2), dtype=np.int64)
        cm2[1, 1] = cm[positive_class, positive_class]
        cm2[1, 0] = cm[positive_class, other_idx].sum()
        cm2[0, 1] = cm[other_idx, positive_class].sum()
        cm2[0, 0] = cm[np.ix_(other_idx, other_idx)].sum()
        return cm2

    def plot_confusion_matrix_binary(self, model_name, debug=False, debug_event_id=None):
        """Binary confusion matrix."""
        cm = self.compute_confusion_matrix(model_name, debug, debug_event_id)
        if cm is None:
            return

        cm2 = self._collapse_confusion_matrix_binary(cm)
        cm2_norm = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]
        cm2_norm = np.nan_to_num(cm2_norm)

        labels = ['Rest', 'Same Cluster']
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm2_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Binary Confusion Matrix - {model_name}')

        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['binary_metrics'], f"{model_name}_confusion_binary.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_loss_curves(self, model_name, metrics):
        """Generate training and testing loss curves."""
        train_losses = metrics.get('train_loss', metrics.get('train_losses', []))
        val_losses = metrics.get('val_loss', metrics.get('val_losses', []))
        test_losses = metrics.get('test_loss', metrics.get('test_losses', []))
        test_epochs = metrics.get('test_epochs', [])
        
        if not train_losses and not val_losses and not test_losses:
            print(f"      ⚠️ No loss data available for {model_name}")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1 = axes[0]
        if train_losses:
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            if val_losses and len(val_losses) == len(train_losses):
                ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title(f'Training and Validation Loss - {model_name}', 
                         fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            best_epoch = metrics.get('best_epoch', 0)
            if isinstance(best_epoch, int) and best_epoch > 0 and best_epoch <= len(train_losses):
                ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                           label=f'Best Epoch: {best_epoch}')
                ax1.legend()
        
        ax2 = axes[1]
        if test_losses:
            if not test_epochs or len(test_epochs) != len(test_losses):
                test_epochs = range(1, len(test_losses) + 1)
            ax2.plot(test_epochs, test_losses, 'g-', label='Test Loss', 
                    linewidth=2, marker='o', markersize=4)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title(f'Test Loss - {model_name}', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            if test_losses:
                min_idx = np.argmin(test_losses)
                min_loss = test_losses[min_idx]
                min_epoch = test_epochs[min_idx] if test_epochs else min_idx + 1
                ax2.scatter(min_epoch, min_loss, color='red', s=100, zorder=5)
                ax2.annotate(f'Min: {min_loss:.4f}', 
                           xy=(min_epoch, min_loss),
                           xytext=(10, 10), textcoords='offset points',
                           fontweight='bold')
        elif train_losses:
            epochs = range(1, len(train_losses) + 1)
            ax2.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.set_title(f'Training Loss - {model_name}', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.folders['loss_curves'], 
                                f"{model_name}_loss_curves.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      ✓ Loss curves saved")
    
    def generate_all_metrics(self, model_names, debug=False, debug_event_id=None):
        """Generate all metric plots for all models."""
        all_results = []
        
        for model_name in model_names:
            print(f"\n📊 Analyzing metrics for: {model_name}")
            try:
                pkl_path = os.path.join(self.models_dir, f"{model_name}_metrics.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        metrics = pickle.load(f)
                else:
                    metrics = {}
                
                extracted = self.extract_model_metrics(metrics)
                model_args = metrics.get('args', {})
                
                result = {
                    'name': model_name,
                    'architecture': model_args.get('model', 'unknown'),
                    'pretrained': extracted.get('pretrained', False),
                    'pretrain_mask_type': extracted.get('pretrain_mask_type'),
                    'pretrain_mask_ratio': extracted.get('pretrain_mask_ratio'),
                    **extracted
                }
                all_results.append(result)
                
                self.plot_roc_curves(model_name, extracted, debug, debug_event_id)
                self.plot_rejection_curves(model_name, extracted, debug, debug_event_id)
                self.plot_pr_curves(model_name, extracted, debug, debug_event_id)
                self.plot_confusion_matrix(model_name, extracted, debug, debug_event_id)
                self.plot_loss_curves(model_name, metrics)
                self.plot_roc_curves_binary(model_name, debug, debug_event_id)
                self.plot_pr_curves_binary(model_name, debug, debug_event_id)
                self.plot_confusion_matrix_binary(model_name, debug, debug_event_id)
                gc.collect()
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:150]}")
        
        return all_results


# ============================================================================
# COMPONENT 2: ClusterReconstructionValidator (WITH ALL IMPROVEMENTS)
# ============================================================================
class ClusterReconstructionValidator:
    """Validates reconstructed clusters against ground truth."""
    
    def __init__(self, confidence_thresholds=[0.5, 0.7, 0.9], min_edges_per_cluster=1):
        self.confidence_thresholds = confidence_thresholds
        self.min_edges_per_cluster = min_edges_per_cluster
    
    def load_event_edges(self, event_id, parquet_dir, model_name):
        """Load predicted edges for a single event."""
        parquet_path = os.path.join(parquet_dir, f"results_{model_name}.parquet")
        if not os.path.exists(parquet_path):
            return None
        
        try:
            edges_df = pq.read_table(
                parquet_path,
                filters=[("event_id", "=", event_id)]
            ).to_pandas()
            return edges_df
        except Exception as e:
            print(f"  ❌ Error loading edges for event {event_id}: {e}")
            return None
    
    def load_truth_clusters(self, event_id, h5_file_or_path, event_offset=0):
        """Load ground truth cluster assignments."""
        try:
            h5_event_id = event_id - event_offset
            if isinstance(h5_file_or_path, h5py.File):
                cluster_indices = h5_file_or_path["cell/cell_cluster_index"][h5_event_id]
            else:
                with h5py.File(h5_file_or_path, 'r') as f:
                    cluster_indices = f["cell/cell_cluster_index"][h5_event_id]
            return cluster_indices.astype(np.int32)
        except Exception as e:
            print(f"  ❌ Error loading truth clusters for event {event_id}: {e}")
            return None
    
    def union_find_clusters(self, edges_df, confidence_threshold, n_cells=None, 
                           use_not_noise_cut=False):
        """
        Build clusters using union-find.
        
        Args:
            edges_df: DataFrame with predicted edges
            confidence_threshold: Min confidence for edge acceptance
            n_cells: Total number of cells
            use_not_noise_cut: If True, use 1-P(Lone-Lone) as the criterion
        """
        if edges_df is None or len(edges_df) == 0:
            return (np.zeros(n_cells, dtype=np.int64) if n_cells is not None else np.array([])), 0

        if use_not_noise_cut:
            same_cluster_edges = edges_df[
                (1 - edges_df['score_class_0']) >= confidence_threshold
            ]
        else:
            same_cluster_edges = edges_df[
                (edges_df['pred_label'] == 1) &
                (edges_df['confidence'] >= confidence_threshold)
            ]

        if len(same_cluster_edges) == 0:
            return (np.zeros(n_cells, dtype=np.int64) if n_cells is not None else np.array([])), 0

        if n_cells is None:
            n_cells = max(
                same_cluster_edges['source_id'].max(),
                same_cluster_edges['target_id'].max()
            ) + 1

        rows = same_cluster_edges['source_id'].values
        cols = same_cluster_edges['target_id'].values
        data = np.ones(len(same_cluster_edges), dtype=np.int8)

        adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
        adj_matrix = adj_matrix + adj_matrix.T

        n_clusters, labels = connected_components(adj_matrix, directed=False, return_labels=True)
        cluster_sizes = np.bincount(labels, minlength=n_clusters)

        final_labels = np.zeros_like(labels)
        cluster_count = 0
        for i in range(n_clusters):
            if cluster_sizes[i] >= self.min_edges_per_cluster + 1:
                cluster_count += 1
                final_labels[labels == i] = cluster_count

        return final_labels, cluster_count

    def validate_clustering_on_truth(self, edges_df, truth_labels, n_cells):
        """
        Sanity check: Run union-find using TRUE labels.
        """
        if edges_df is None or len(edges_df) == 0:
            return np.zeros(n_cells), 0, None
        
        true_same_edges = edges_df[edges_df['true_label'] == 1]
        
        if len(true_same_edges) == 0:
            return np.zeros(n_cells), 0, None
        
        rows = true_same_edges['source_id'].values
        cols = true_same_edges['target_id'].values
        data = np.ones(len(true_same_edges), dtype=np.int8)
        
        adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
        adj_matrix = adj_matrix + adj_matrix.T
        
        n_clusters, labels = connected_components(adj_matrix, directed=False, return_labels=True)
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        
        final_labels = np.zeros_like(labels)
        cluster_count = 0
        for i in range(n_clusters):
            if cluster_sizes[i] >= 2:
                cluster_count += 1
                final_labels[labels == i] = cluster_count
        
        comparison = self.compute_cluster_metrics(final_labels, truth_labels)
        
        return final_labels, cluster_count, comparison
    
    def compute_cluster_metrics(self, predicted_labels, truth_labels, exclude_mask=None):
        """Match clusters using Hungarian algorithm."""
        if exclude_mask is not None:
            predicted_labels = predicted_labels.copy()
            truth_labels = truth_labels.copy()
            if len(predicted_labels) == len(exclude_mask):
                predicted_labels[exclude_mask] = 0
            if len(truth_labels) == len(exclude_mask):
                truth_labels[exclude_mask] = 0

        pred_clusters = np.unique(predicted_labels[predicted_labels > 0])
        truth_clusters = np.unique(truth_labels[truth_labels > 0])

        n_pred = len(pred_clusters)
        n_truth = len(truth_clusters)

        if n_pred == 0 or n_truth == 0:
            return {
                'n_pred_clusters': n_pred, 'n_truth_clusters': n_truth,
                'n_matched': 0, 'mean_iou': 0.0, 'mean_purity': 0.0,
                'mean_efficiency': 0.0, 'cluster_count_diff': n_pred - n_truth
            }

        iou_matrix = np.zeros((n_pred, n_truth))
        for i, pc in enumerate(pred_clusters):
            pred_cells = set(np.where(predicted_labels == pc)[0])
            for j, tc in enumerate(truth_clusters):
                truth_cells = set(np.where(truth_labels == tc)[0])
                intersection = len(pred_cells & truth_cells)
                union = len(pred_cells | truth_cells)
                iou_matrix[i, j] = intersection / union if union > 0 else 0.0

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_ious = iou_matrix[row_ind, col_ind]

        purities, efficiencies = [], []
        for i, j in zip(row_ind, col_ind):
            pred_cells = set(np.where(predicted_labels == pred_clusters[i])[0])
            truth_cells = set(np.where(truth_labels == truth_clusters[j])[0])
            intersection = len(pred_cells & truth_cells)
            purities.append(intersection / len(pred_cells) if len(pred_cells) > 0 else 0.0)
            efficiencies.append(intersection / len(truth_cells) if len(truth_cells) > 0 else 0.0)

        good_matches = matched_ious >= 0.1
        n_good_matches = good_matches.sum()

        if n_good_matches > 0:
            mean_iou = np.mean(matched_ious[good_matches])
            mean_purity = np.mean(np.array(purities)[good_matches])
            mean_efficiency = np.mean(np.array(efficiencies)[good_matches])
        else:
            mean_iou = mean_purity = mean_efficiency = 0.0

        return {
            'n_pred_clusters': n_pred, 'n_truth_clusters': n_truth,
            'n_matched': n_good_matches, 'mean_iou': float(mean_iou),
            'mean_purity': float(mean_purity), 'mean_efficiency': float(mean_efficiency),
            'cluster_count_diff': n_pred - n_truth
        }

    def get_cluster_match_info(self, predicted_labels, truth_labels, iou_threshold=0.1):
        """
        Same Hungarian-IoU matching as compute_cluster_metrics, but returns
        per-predicted-cluster match status — used to flag 'extra' (spurious,
        unmatched) predicted clusters for visualization.
        
        Returns:
            dict: {pred_cluster_id: {'matched': bool, 'truth_cluster': int or None, 'iou': float}}
        """
        pred_clusters = np.unique(predicted_labels[predicted_labels > 0])
        truth_clusters = np.unique(truth_labels[truth_labels > 0])
        
        match_info = {
            int(pc): {'matched': False, 'truth_cluster': None, 'iou': 0.0} 
            for pc in pred_clusters
        }
        
        if len(pred_clusters) == 0 or len(truth_clusters) == 0:
            return match_info
        
        iou_matrix = np.zeros((len(pred_clusters), len(truth_clusters)))
        for i, pc in enumerate(pred_clusters):
            pred_cells = set(np.where(predicted_labels == pc)[0])
            for j, tc in enumerate(truth_clusters):
                truth_cells = set(np.where(truth_labels == tc)[0])
                intersection = len(pred_cells & truth_cells)
                union = len(pred_cells | truth_cells)
                iou_matrix[i, j] = intersection / union if union > 0 else 0.0
        
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        for i, j in zip(row_ind, col_ind):
            iou = iou_matrix[i, j]
            if iou >= iou_threshold:
                pred_cluster_id = int(pred_clusters[i])
                truth_cluster_id = int(truth_clusters[j])
                match_info[pred_cluster_id] = {
                    'matched': True,
                    'truth_cluster': truth_cluster_id,
                    'iou': float(iou)
                }
        
        return match_info

    def match_clusters_by_delta_r(self, predicted_labels, truth_labels, cell_eta, cell_phi,
                                   cell_energy=None, delta_r_threshold=0.05):
        """
        Match each predicted cluster to its geometrically closest truth
        cluster centroid using ΔR = sqrt(Δη² + Δφ²).
        
        Args:
            predicted_labels: array of predicted cluster assignments
            truth_labels: array of truth cluster assignments
            cell_eta, cell_phi: cell coordinates
            cell_energy: cell energies (optional, for weighting)
            delta_r_threshold: maximum ΔR for matching
            
        Returns:
            tuple: (results_list, summary_dict)
        """
        pred_clusters = np.unique(predicted_labels[predicted_labels > 0])
        truth_clusters = np.unique(truth_labels[truth_labels > 0])
        
        def compute_centroid(labels, cluster_id):
            """Compute energy-weighted centroid for a cluster."""
            cells = np.where(labels == cluster_id)[0]
            if len(cells) == 0:
                return None
            
            if cell_energy is not None:
                weights = np.clip(cell_energy[cells], 0, None)
                if weights.sum() <= 0:
                    weights = np.ones(len(cells))
            else:
                weights = np.ones(len(cells))
            
            eta_centroid = np.average(cell_eta[cells], weights=weights)
            sin_mean = np.average(np.sin(cell_phi[cells]), weights=weights)
            cos_mean = np.average(np.cos(cell_phi[cells]), weights=weights)
            phi_centroid = np.arctan2(sin_mean, cos_mean)
            
            return eta_centroid, phi_centroid
        
        def compute_delta_r(eta1, phi1, eta2, phi2):
            """Compute ΔR between two points."""
            deta = eta1 - eta2
            dphi = np.abs(phi1 - phi2)
            dphi = np.minimum(dphi, 2 * np.pi - dphi)
            return np.sqrt(deta**2 + dphi**2)
        
        # Compute truth centroids
        truth_centroids = {}
        for tc in truth_clusters:
            centroid = compute_centroid(truth_labels, tc)
            if centroid is not None:
                truth_centroids[int(tc)] = centroid
        
        # Match each predicted cluster
        results = []
        for pc in pred_clusters:
            pred_centroid = compute_centroid(predicted_labels, pc)
            if pred_centroid is None:
                continue
            
            eta_p, phi_p = pred_centroid
            
            # Find closest truth cluster
            best_truth = None
            best_dr = np.inf
            
            for tc, (eta_t, phi_t) in truth_centroids.items():
                dr = compute_delta_r(eta_p, phi_p, eta_t, phi_t)
                if dr < best_dr:
                    best_dr = dr
                    best_truth = tc
            
            entry = {
                'pred_cluster': int(pc),
                'matched': best_dr <= delta_r_threshold,
                'matched_truth_cluster': best_truth if best_dr <= delta_r_threshold else None,
                'delta_r': float(best_dr) if best_truth is not None else None,
            }
            
            # Compute quality metrics if matched
            if entry['matched']:
                pred_cells = set(np.where(predicted_labels == pc)[0])
                truth_cells = set(np.where(truth_labels == best_truth)[0])
                intersection = len(pred_cells & truth_cells)
                union = len(pred_cells | truth_cells)
                
                entry.update({
                    'iou': intersection / union if union > 0 else 0.0,
                    'purity': intersection / len(pred_cells) if len(pred_cells) > 0 else 0.0,
                    'efficiency': intersection / len(truth_cells) if len(truth_cells) > 0 else 0.0,
                })
            else:
                entry.update({
                    'iou': 0.0,
                    'purity': 0.0,
                    'efficiency': 0.0,
                })
            
            results.append(entry)
        
        # Summary statistics
        matched_truths = {r['matched_truth_cluster'] for r in results if r['matched']}
        n_extra = sum(1 for r in results if not r['matched'])
        n_missed = len(truth_clusters) - len(matched_truths)
        
        summary = {
            'n_pred_clusters': len(pred_clusters),
            'n_truth_clusters': len(truth_clusters),
            'n_matched': len(results) - n_extra,
            'n_extra_clusters': n_extra,
            'n_missed_truth_clusters': n_missed,
        }
        
        # Add summary metrics if there are matches
        if results:
            matched_results = [r for r in results if r['matched']]
            if matched_results:
                summary['mean_iou_matched'] = np.mean([r['iou'] for r in matched_results])
                summary['mean_purity_matched'] = np.mean([r['purity'] for r in matched_results])
                summary['mean_efficiency_matched'] = np.mean([r['efficiency'] for r in matched_results])
        
        return results, summary

    def plot_delta_r_matching_summary(self, delta_r_df, output_folder, model_name):
        """Generate comprehensive visualizations for ΔR matching results."""
        if delta_r_df is None or delta_r_df.empty:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'ΔR-Based Cluster Matching Summary - {model_name}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Number of extra clusters per event
        ax = axes[0, 0]
        ax.hist(delta_r_df['n_extra_clusters'], bins=20, alpha=0.7, 
                color='red', edgecolor='black')
        ax.axvline(x=delta_r_df['n_extra_clusters'].mean(), color='darkred', 
                   linestyle='--', label=f"Mean: {delta_r_df['n_extra_clusters'].mean():.1f}")
        ax.set_xlabel('Number of Extra Clusters')
        ax.set_ylabel('Number of Events')
        ax.set_title('Extra (Spurious) Clusters per Event')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Number of missed truth clusters per event
        ax = axes[0, 1]
        ax.hist(delta_r_df['n_missed_truth_clusters'], bins=20, alpha=0.7, 
                color='orange', edgecolor='black')
        ax.axvline(x=delta_r_df['n_missed_truth_clusters'].mean(), color='darkorange', 
                   linestyle='--', label=f"Mean: {delta_r_df['n_missed_truth_clusters'].mean():.1f}")
        ax.set_xlabel('Number of Missed Truth Clusters')
        ax.set_ylabel('Number of Events')
        ax.set_title('Missed Truth Clusters per Event')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Matching efficiency
        if 'n_matched' in delta_r_df.columns and 'n_truth_clusters' in delta_r_df.columns:
            match_efficiency = delta_r_df['n_matched'] / delta_r_df['n_truth_clusters'].clip(lower=1)
            ax = axes[0, 2]
            ax.hist(match_efficiency, bins=20, alpha=0.7, 
                    color='blue', edgecolor='black')
            ax.axvline(x=match_efficiency.mean(), color='darkblue', 
                       linestyle='--', label=f"Mean: {match_efficiency.mean():.2f}")
            ax.set_xlabel('Matching Efficiency (Matched/Truth)')
            ax.set_ylabel('Number of Events')
            ax.set_title('Cluster Matching Efficiency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. IoU distribution for matched clusters
        if 'mean_iou_matched' in delta_r_df.columns:
            ax = axes[1, 0]
            ax.hist(delta_r_df['mean_iou_matched'].dropna(), bins=20, alpha=0.7, 
                    color='green', edgecolor='black')
            ax.axvline(x=delta_r_df['mean_iou_matched'].mean(), color='darkgreen', 
                       linestyle='--', label=f"Mean: {delta_r_df['mean_iou_matched'].mean():.2f}")
            ax.set_xlabel('Mean IoU (Matched Clusters)')
            ax.set_ylabel('Number of Events')
            ax.set_title('IoU for ΔR-Matched Clusters')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 5. Purity vs Efficiency scatter
        if 'mean_purity_matched' in delta_r_df.columns and 'mean_efficiency_matched' in delta_r_df.columns:
            ax = axes[1, 1]
            scatter = ax.scatter(delta_r_df['mean_purity_matched'], 
                                delta_r_df['mean_efficiency_matched'],
                                c=delta_r_df['n_extra_clusters'], 
                                cmap='RdYlBu_r', alpha=0.6, s=50)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect')
            ax.set_xlabel('Mean Purity')
            ax.set_ylabel('Mean Efficiency')
            ax.set_title('Purity vs Efficiency (color = extra clusters)')
            plt.colorbar(scatter, ax=ax, label='Extra Clusters')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax = axes[1, 2]
        metrics = [
            ('Extra Clusters', delta_r_df['n_extra_clusters'].mean(), 'red'),
            ('Missed Truth', delta_r_df['n_missed_truth_clusters'].mean(), 'orange'),
            ('Matched Clusters', delta_r_df['n_matched'].mean(), 'blue'),
            ('Mean IoU', delta_r_df['mean_iou_matched'].mean(), 'green'),
            ('Mean Purity', delta_r_df['mean_purity_matched'].mean(), 'purple'),
            ('Mean Efficiency', delta_r_df['mean_efficiency_matched'].mean(), 'brown'),
        ]
        names = [m[0] for m in metrics]
        values = [m[1] for m in metrics]
        colors = [m[2] for m in metrics]
        
        bars = ax.bar(names, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Mean Value')
        ax.set_title('Summary Statistics')
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(output_folder, f"{model_name}_delta_r_summary.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ ΔR matching visualization saved: {os.path.basename(save_path)}")

    def plot_delta_r_event_details(self, delta_r_results, event_id, output_folder, model_name):
        """Plot detailed ΔR matching results for a single event."""
        if not delta_r_results:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'ΔR Matching Details - Event {event_id} - {model_name}', 
                     fontsize=14, fontweight='bold')
        
        # 1. ΔR distribution
        ax = axes[0]
        delta_rs = [r['delta_r'] for r in delta_r_results if r['delta_r'] is not None]
        if delta_rs:
            ax.hist(delta_rs, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(x=0.05, color='red', linestyle='--', label='Threshold (0.05)')
            ax.set_xlabel('ΔR')
            ax.set_ylabel('Count')
            ax.set_title('ΔR Distribution for Matches')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. IoU for matched clusters
        ax = axes[1]
        matched_ious = [r['iou'] for r in delta_r_results if r['matched']]
        if matched_ious:
            ax.hist(matched_ious, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(x=np.mean(matched_ious), color='darkgreen', 
                       linestyle='--', label=f"Mean: {np.mean(matched_ious):.2f}")
            ax.set_xlabel('IoU')
            ax.set_ylabel('Count')
            ax.set_title('IoU for ΔR-Matched Clusters')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Purity vs Efficiency
        ax = axes[2]
        matched = [r for r in delta_r_results if r['matched']]
        if matched:
            purities = [r['purity'] for r in matched]
            efficiencies = [r['efficiency'] for r in matched]
            ax.scatter(purities, efficiencies, alpha=0.6, s=50)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_xlabel('Purity')
            ax.set_ylabel('Efficiency')
            ax.set_title('Purity vs Efficiency (Matched)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_folder, f"{model_name}_event{event_id}_delta_r_details.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_efficiency_eta_phi_truth_check(self, edges_df, truth_labels, cell_eta, cell_phi,
                                             n_cells, output_folder, model_name, event_id):
        """
        Same eta/phi efficiency profile as plot_efficiency_eta_phi, but built
        from validate_clustering_on_truth (pure true_label==1 edges) instead
        of model predictions.
        """
        pred_labels, n_clusters, comparison = self.validate_clustering_on_truth(
            edges_df, truth_labels, n_cells
        )
        
        self.plot_efficiency_eta_phi(
            pred_labels, truth_labels, cell_eta, cell_phi,
            output_folder, f"{model_name}_TRUTH_CHECK", event_id
        )
        
        if comparison:
            print(f"  📊 Truth check (Event {event_id}): "
                  f"IoU={comparison.get('mean_iou', 0):.3f}, "
                  f"Purity={comparison.get('mean_purity', 0):.3f}, "
                  f"Efficiency={comparison.get('mean_efficiency', 0):.3f}")
        
        return comparison

    def compute_truth_recovery(self, predicted_labels, truth_labels):
        """For each truth cluster, what fraction of cells are recovered."""
        truth_clusters = np.unique(truth_labels[truth_labels > 0])
        results = []
        
        for tc in truth_clusters:
            truth_cells = np.where(truth_labels == tc)[0]
            pred_ids = predicted_labels[truth_cells]
            pred_ids = pred_ids[pred_ids > 0]
            
            if len(pred_ids) == 0:
                results.append({'truth_cluster': tc, 'n_truth_cells': len(truth_cells),
                               'recovery': 0.0, 'best_pred_cluster': None})
                continue
            
            best_pred = np.bincount(pred_ids).argmax()
            n_recovered = (pred_ids == best_pred).sum()
            results.append({'truth_cluster': tc, 'n_truth_cells': len(truth_cells),
                           'recovery': n_recovered / len(truth_cells),
                           'best_pred_cluster': best_pred})
        
        return results

    def compute_energy_weighted_metrics(self, predicted_labels, truth_labels, cell_energy):
        """Energy-weighted purity and efficiency."""
        truth_clusters = np.unique(truth_labels[truth_labels > 0])
        pred_clusters = np.unique(predicted_labels[predicted_labels > 0])
        
        results = []
        
        if len(pred_clusters) == 0 or len(truth_clusters) == 0:
            return results
        
        iou_matrix = np.zeros((len(pred_clusters), len(truth_clusters)))
        for i, pc in enumerate(pred_clusters):
            pred_cells = set(np.where(predicted_labels == pc)[0])
            for j, tc in enumerate(truth_clusters):
                truth_cells = set(np.where(truth_labels == tc)[0])
                intersection = len(pred_cells & truth_cells)
                union = len(pred_cells | truth_cells)
                iou_matrix[i, j] = intersection / union if union > 0 else 0.0
        
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        for i, j in zip(row_ind, col_ind):
            pred_cells = np.where(predicted_labels == pred_clusters[i])[0]
            truth_cells = np.where(truth_labels == truth_clusters[j])[0]
            intersection = set(pred_cells) & set(truth_cells)
            
            pred_energy = cell_energy[pred_cells].sum()
            truth_energy = cell_energy[truth_cells].sum()
            intersection_energy = cell_energy[list(intersection)].sum()
            
            energy_purity = intersection_energy / pred_energy if pred_energy > 0 else 0
            energy_efficiency = intersection_energy / truth_energy if truth_energy > 0 else 0
            
            energy_diff_mean = np.abs(pred_energy - truth_energy)
            energy_diff_rms = np.sqrt((pred_energy - truth_energy)**2)
            
            results.append({
                'pred_cluster': int(pred_clusters[i]),
                'truth_cluster': int(truth_clusters[j]),
                'energy_purity': float(energy_purity),
                'energy_efficiency': float(energy_efficiency),
                'energy_diff_mean': float(energy_diff_mean),
                'energy_diff_rms': float(energy_diff_rms),
            })
        
        return results

    def union_find_clusters_cc_corrected(self, edges_df, confidence_threshold, n_cells=None):
        """Option A: Oracle-corrected for CC confusion."""
        if edges_df is None or len(edges_df) == 0:
            return (np.zeros(n_cells, dtype=np.int64) if n_cells is not None else np.array([])), 0

        same_cluster_edges = edges_df[
            (
                (edges_df['pred_label'] == 1) |
                ((edges_df['true_label'] == 1) & (edges_df['pred_label'] == 4))
            ) &
            (edges_df['confidence'] >= confidence_threshold)
        ]

        if len(same_cluster_edges) == 0:
            return (np.zeros(n_cells, dtype=np.int64) if n_cells is not None else np.array([])), 0

        if n_cells is None:
            n_cells = max(
                same_cluster_edges['source_id'].max(),
                same_cluster_edges['target_id'].max()
            ) + 1

        rows = same_cluster_edges['source_id'].values
        cols = same_cluster_edges['target_id'].values
        data = np.ones(len(same_cluster_edges), dtype=np.int8)

        adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
        adj_matrix = adj_matrix + adj_matrix.T

        n_clusters, labels = connected_components(adj_matrix, directed=False, return_labels=True)
        cluster_sizes = np.bincount(labels, minlength=n_clusters)

        final_labels = np.zeros_like(labels)
        cluster_count = 0
        for i in range(n_clusters):
            if cluster_sizes[i] >= self.min_edges_per_cluster + 1:
                cluster_count += 1
                final_labels[labels == i] = cluster_count

        return final_labels, cluster_count

    def compute_cell_cc_fraction(self, edges_df, n_cells):
        """Per cell, fraction of edges whose TRUE label is 4."""
        cc_fraction = np.zeros(n_cells)
        if edges_df is None or len(edges_df) == 0:
            return cc_fraction

        stacked = pd.concat([
            edges_df[['source_id', 'true_label']].rename(columns={'source_id': 'cell'}),
            edges_df[['target_id', 'true_label']].rename(columns={'target_id': 'cell'}),
        ], ignore_index=True)

        counts = stacked.groupby('cell')['true_label'].agg(
            total='count',
            cc_count=lambda s: (s == 4).sum()
        )
        valid = counts.index[counts.index < n_cells]
        cc_fraction[valid] = (counts.loc[valid, 'cc_count'] / counts.loc[valid, 'total']).values
        return cc_fraction

    def plot_iou_distribution(self, all_ious, output_folder, model_name):
        """Plot histogram of IoU values."""
        if not all_ious:
            return
        
        all_ious = np.array(all_ious)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_ious, bins=50, range=(0, 1), alpha=0.7, edgecolor='black')
        ax.axvline(x=0.1, color='red', linestyle='--', label='Current threshold (0.1)')
        
        suggested = np.percentile(all_ious, 25)
        ax.axvline(x=suggested, color='green', linestyle='--', 
                   label=f'Suggested threshold ({suggested:.2f})')
        
        ax.set_xlabel('IoU')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of IoU Values - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{model_name}_iou_distribution.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()

    def plot_efficiency_eta_phi(self, predicted_labels, truth_labels, 
                                cell_eta, cell_phi, output_folder, model_name, event_id):
        """Plot per-cell recovery efficiency vs η and φ."""
        cell_efficiency = np.zeros(len(truth_labels))
        
        for tc in np.unique(truth_labels[truth_labels > 0]):
            truth_cells = np.where(truth_labels == tc)[0]
            pred_ids = predicted_labels[truth_cells]
            cell_efficiency[truth_cells] = (pred_ids > 0).astype(float)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Eta profile
        ax1 = axes[0]
        eta_bins = np.linspace(-3.5, 3.5, 50)
        eta_means, eta_centers = [], []
        for i in range(len(eta_bins)-1):
            mask = (cell_eta >= eta_bins[i]) & (cell_eta < eta_bins[i+1])
            mask &= (truth_labels > 0)
            if mask.sum() > 10:
                eta_means.append(cell_efficiency[mask].mean())
                eta_centers.append((eta_bins[i] + eta_bins[i+1]) / 2)
        
        ax1.plot(eta_centers, eta_means, 'bo', markersize=4)
        ax1.axhline(y=1.0, color='green', linestyle='--', label='Perfect (1.0)')
        ax1.set_xlabel('η')
        ax1.set_ylabel('Cell Efficiency')
        ax1.set_title(f'Cluster Efficiency vs η (Event {event_id})')
        ax1.set_ylim([0, 1.05])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Phi profile
        ax2 = axes[1]
        phi_bins = np.linspace(-3.5, 3.5, 50)
        phi_means, phi_centers = [], []
        for i in range(len(phi_bins)-1):
            mask = (cell_phi >= phi_bins[i]) & (cell_phi < phi_bins[i+1])
            mask &= (truth_labels > 0)
            if mask.sum() > 10:
                phi_means.append(cell_efficiency[mask].mean())
                phi_centers.append((phi_bins[i] + phi_bins[i+1]) / 2)
        
        ax2.plot(phi_centers, phi_means, 'ro', markersize=4)
        ax2.axhline(y=1.0, color='green', linestyle='--', label='Perfect (1.0)')
        ax2.set_xlabel('φ')
        ax2.set_ylabel('Cell Efficiency')
        ax2.set_title(f'Cluster Efficiency vs φ (Event {event_id})')
        ax2.set_ylim([0, 1.05])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{model_name}_event{event_id}_efficiency_eta_phi.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()

    def plot_energy_metrics_visualization(self, energy_df, output_folder, model_name, event_id):
        """Generate 4-panel visualization of energy-weighted metrics."""
        if energy_df is None or energy_df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Energy Purity vs Energy Efficiency scatter
        ax = axes[0, 0]
        ax.scatter(energy_df['energy_purity'], energy_df['energy_efficiency'], 
                   alpha=0.7, s=50, c='blue', edgecolors='black')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal line')
        ax.set_xlabel('Energy Purity')
        ax.set_ylabel('Energy Efficiency')
        ax.set_title(f'Energy Purity vs Efficiency - {model_name} (Event {event_id})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # 2. Histograms of energy purity and efficiency
        ax = axes[0, 1]
        ax.hist(energy_df['energy_purity'], bins=20, alpha=0.5, label='Purity', color='blue')
        ax.hist(energy_df['energy_efficiency'], bins=20, alpha=0.5, label='Efficiency', color='green')
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Energy Purity and Efficiency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Energy difference per cluster pair
        ax = axes[1, 0]
        pairs = np.arange(len(energy_df))
        ax.bar(pairs - 0.2, energy_df['energy_diff_mean'], 0.4, label='Mean Diff', color='red', alpha=0.7)
        ax.bar(pairs + 0.2, energy_df['energy_diff_rms'], 0.4, label='RMS Diff', color='purple', alpha=0.7)
        ax.set_xlabel('Matched Cluster Pair Index')
        ax.set_ylabel('Energy Difference')
        ax.set_title('Energy Difference per Matched Pair')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        ax = axes[1, 1]
        metrics_summary = {
            'Energy\nPurity': energy_df['energy_purity'].mean(),
            'Energy\nEfficiency': energy_df['energy_efficiency'].mean(),
            'Mean\nEnergy Diff': energy_df['energy_diff_mean'].mean(),
            'RMS\nEnergy Diff': energy_df['energy_diff_rms'].mean(),
        }
        names = list(metrics_summary.keys())
        values = list(metrics_summary.values())
        colors = ['blue', 'green', 'red', 'purple']
        ax.bar(names, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Mean Value')
        ax.set_title('Summary Statistics')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(output_folder, f"{model_name}_event{event_id}_energy_visualization.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Energy visualization saved: {os.path.basename(save_path)}")

    def validate_event_ablation(self, event_id, parquet_dir, model_name, h5_file_or_path,
                                 event_offset=0, cc_heavy_threshold=0.5):
        """Run baseline + Option A + Option B for one event."""
        edges_df = self.load_event_edges(event_id, parquet_dir, model_name)
        truth_labels = self.load_truth_clusters(event_id, h5_file_or_path, event_offset)

        if edges_df is None or truth_labels is None:
            return {'event_id': event_id, 'error': 'Failed to load data'}

        n_cells = len(truth_labels)
        cc_fraction = self.compute_cell_cc_fraction(edges_df, n_cells)
        cc_heavy_mask = (truth_labels > 0) & (cc_fraction >= cc_heavy_threshold)

        results = {'event_id': event_id, 'n_cc_heavy_cells': int(cc_heavy_mask.sum())}

        for threshold in self.confidence_thresholds:
            pred_baseline, _ = self.union_find_clusters(edges_df, threshold, n_cells)
            metrics_baseline = self.compute_cluster_metrics(pred_baseline, truth_labels)

            pred_ccfix, _ = self.union_find_clusters_cc_corrected(edges_df, threshold, n_cells)
            metrics_ccfix = self.compute_cluster_metrics(pred_ccfix, truth_labels)

            metrics_excl = self.compute_cluster_metrics(pred_baseline, truth_labels, exclude_mask=cc_heavy_mask)

            results[f'threshold_{threshold}'] = {
                'baseline': metrics_baseline,
                'cc_corrected': metrics_ccfix,
                'cc_excluded': metrics_excl,
            }

        return results

    def validate_all_events_ablation(self, event_ids, parquet_dir, model_name, h5_file_or_path,
                                      save_path=None, event_offset=0, cc_heavy_threshold=0.5):
        """Run ablation across many events."""
        rows = []
        for event_id in event_ids:
            result = self.validate_event_ablation(
                event_id, parquet_dir, model_name, h5_file_or_path,
                event_offset, cc_heavy_threshold
            )
            if 'error' in result:
                continue
            for threshold in self.confidence_thresholds:
                variants = result.get(f'threshold_{threshold}', {})
                for variant_name, metrics in variants.items():
                    rows.append({
                        'event_id': result['event_id'],
                        'confidence_threshold': threshold,
                        'variant': variant_name,
                        'n_cc_heavy_cells': result['n_cc_heavy_cells'],
                        **metrics
                    })

        results_df = pd.DataFrame(rows)
        if save_path:
            results_df.to_parquet(save_path)
        return results_df

    def plot_ablation_comparison(self, ablation_df, output_folder, model_name):
        """Grouped bar chart for ablation comparison."""
        if ablation_df is None or ablation_df.empty:
            return

        variants = ['baseline', 'cc_corrected', 'cc_excluded']
        variant_labels = ['As-Predicted', 'Option A: CC-Corrected', 'Option B: CC-Heavy Excluded']
        variant_colors = ['#3498db', '#2ecc71', '#e67e22']
        thresholds = sorted(ablation_df['confidence_threshold'].unique())
        metrics_to_plot = ['mean_iou', 'mean_purity', 'mean_efficiency']

        fig, axes = plt.subplots(1, len(thresholds), figsize=(6 * len(thresholds), 6), sharey=True)
        if len(thresholds) == 1:
            axes = [axes]

        for ax, threshold in zip(axes, thresholds):
            sub = ablation_df[ablation_df['confidence_threshold'] == threshold]
            x = np.arange(len(metrics_to_plot))
            width = 0.25

            for i, (variant, label, color) in enumerate(zip(variants, variant_labels, variant_colors)):
                variant_sub = sub[sub['variant'] == variant]
                means = [variant_sub[m].mean() if not variant_sub.empty else 0 for m in metrics_to_plot]
                ax.bar(x + (i - 1) * width, means, width, label=label, color=color, alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(['Mean IoU', 'Mean Purity', 'Mean Efficiency'])
            ax.set_ylim([0, 1])
            ax.set_title(f'Threshold = {threshold}')
            ax.grid(True, alpha=0.3, axis='y')
            if ax is axes[0]:
                ax.set_ylabel('Score')

        axes[-1].legend(loc='upper right', fontsize=8)
        fig.suptitle(f'CC-Confusion Ablation: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{model_name}_cc_ablation.png"),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def validate_event(self, event_id, parquet_dir, model_name, h5_file_or_path, 
                      event_offset=0, use_not_noise_cut=False):
        """Validate single event."""
        edges_df = self.load_event_edges(event_id, parquet_dir, model_name)
        truth_labels = self.load_truth_clusters(event_id, h5_file_or_path, event_offset)
        
        if edges_df is None or truth_labels is None:
            return {'event_id': event_id, 'error': 'Failed to load data'}
        
        n_cells = len(truth_labels)
        results = {'event_id': event_id}
        
        # Sanity check: validate clustering on truth
        truth_pred, truth_n_clusters, truth_comparison = self.validate_clustering_on_truth(
            edges_df, truth_labels, n_cells
        )
        results['truth_validation'] = truth_comparison
        results['truth_n_clusters_recovered'] = truth_n_clusters
        
        for threshold in self.confidence_thresholds:
            pred_labels, n_clusters = self.union_find_clusters(
                edges_df, threshold, n_cells, use_not_noise_cut=use_not_noise_cut
            )
            metrics = self.compute_cluster_metrics(pred_labels, truth_labels)
            metrics.update({'threshold': threshold, 'n_predicted_clusters_total': n_clusters})
            results[f'threshold_{threshold}'] = metrics
        
        return results
    
    def validate_all_events(self, event_ids, parquet_dir, model_name, h5_file_or_path, 
                           save_path=None, event_offset=0, use_not_noise_cut=False,
                           output_folder=None):
        """Validate multiple events with all improvements."""
        all_results = []
        all_ious = []
        
        for event_id in event_ids:
            result = self.validate_event(
                event_id, parquet_dir, model_name, h5_file_or_path, 
                event_offset, use_not_noise_cut
            )
            all_results.append(result)
            
            for threshold in self.confidence_thresholds:
                metrics = result.get(f'threshold_{threshold}', {})
                if metrics and metrics.get('mean_iou', 0) > 0:
                    all_ious.append(metrics['mean_iou'])
        
        rows = []
        for result in all_results:
            if 'error' in result:
                continue
            for threshold in self.confidence_thresholds:
                metrics = result.get(f'threshold_{threshold}', {})
                if metrics:
                    row = {'event_id': result['event_id'], 
                           'confidence_threshold': threshold, **metrics}
                    rows.append(row)
        
        results_df = pd.DataFrame(rows)
        if save_path:
            results_df.to_parquet(save_path)
        
        if output_folder:
            self.plot_iou_distribution(all_ious, output_folder, model_name)
        
        return results_df

# ============================================================================
# COMPONENT 3: EventVisualizer
# ============================================================================
class EventVisualizer:
    """Generates PDF pages with layer groups, masking, and cluster panels."""
    
    def __init__(self, output_dir, h5_path, use_improved_styling=True):
        self.output_dir = output_dir
        self.h5_path = h5_path
        self.use_improved_styling = use_improved_styling
        self.masking_engine = None
        self._n_total_cells = None
        os.makedirs(output_dir, exist_ok=True)
    
    def set_masking_engine(self, engine):
        """Set masking inference engine."""
        self.masking_engine = engine
    
    def _get_total_n_cells(self, h5f=None):
        """True total detector cell count from HDF5 array shape."""
        if self._n_total_cells is None:
            if h5f is not None:
                self._n_total_cells = h5f['cell/cell_eta'].shape[1]
            else:
                with h5py.File(self.h5_path, 'r') as f:
                    self._n_total_cells = f['cell/cell_eta'].shape[1]
        return self._n_total_cells

    def load_event_data(self, event_id, edge_file_path=None, event_offset=0, h5f=None):
        """Load event cells and edges."""
        h5_event_id = event_id - event_offset
        
        should_close = False
        if h5f is None:
            h5f = h5py.File(self.h5_path, 'r')
            should_close = True
        
        try:
            cell_data = {
                "eta": h5f["cell/cell_eta"][h5_event_id].astype(np.float32),
                "phi": h5f["cell/cell_phi"][h5_event_id].astype(np.float32),
                "subCalo": h5f["cell/cell_subCalo"][h5_event_id].astype(np.int8),
                "layer": h5f["cell/cell_sampling"][h5_event_id].astype(np.int8),
                "energy": h5f["cell/cell_e"][h5_event_id].astype(np.float32),
                "cluster_index": h5f["cell/cell_cluster_index"][h5_event_id].astype(np.int16),
            }
        finally:
            if should_close:
                h5f.close()
        
        df = pd.DataFrame(cell_data)
        df['energy'] = df['energy'].clip(lower=0.0)
        df = df[np.isfinite(df[["eta", "phi"]]).all(axis=1)].copy()
        df = df[
            (df['eta'] >= ETA_RANGE[0]) & (df['eta'] <= ETA_RANGE[1]) &
            (df['phi'] >= PHI_RANGE[0]) & (df['phi'] <= PHI_RANGE[1])
        ].copy()
        
        df['truth_label'] = 0
        df['pred_label'] = -1
        df['truth_label_priority'] = 0
        df['pred_label_priority'] = -1
        
        edges_df = pd.DataFrame()
        if edge_file_path and os.path.exists(edge_file_path):
            try:
                edges_df = pq.read_table(
                    edge_file_path,
                    filters=[("event_id", "=", event_id)]
                ).to_pandas()
            except:
                pass
        
        if not edges_df.empty:
            self._derive_cell_labels_from_edges(df, edges_df)
        
        return df, edges_df
    
    def _derive_cell_labels_from_edges(self, df_cells, df_edges):
        """Derive cell labels from edge predictions."""
        cell_truth = {pos: {i: 0 for i in range(5)} for pos in df_cells.index}
        cell_pred = {pos: {i: 0 for i in range(5)} for pos in df_cells.index}
        
        truth_col = 'true_label' if 'true_label' in df_edges.columns else 'truth_label'
        pred_col = 'pred_label' if 'pred_label' in df_edges.columns else 'prediction'
        src_col = 'source_id' if 'source_id' in df_edges.columns else 'source'
        tgt_col = 'target_id' if 'target_id' in df_edges.columns else 'target'
        
        for _, e in df_edges.iterrows():
            src = e[src_col]
            tgt = e[tgt_col]
            t = e[truth_col]
            p = e[pred_col]
            
            if src in cell_truth and t in cell_truth[src]:
                cell_truth[src][t] += 1
                if p in cell_pred[src]:
                    cell_pred[src][p] += 1
            if tgt in cell_truth and t in cell_truth[tgt]:
                cell_truth[tgt][t] += 1
                if p in cell_pred[tgt]:
                    cell_pred[tgt][p] += 1
        
        for idx in df_cells.index:
            if idx in cell_truth and sum(cell_truth[idx].values()) > 0:
                df_cells.loc[idx, 'truth_label'] = max(cell_truth[idx], 
                                                        key=cell_truth[idx].get)
                present_labels = [k for k, v in cell_truth[idx].items() if v > 0]
                best_label = max(present_labels, key=lambda x: PRIORITY_LOOKUP[x])
                df_cells.loc[idx, 'truth_label_priority'] = best_label
            
            if idx in cell_pred and sum(cell_pred[idx].values()) > 0:
                df_cells.loc[idx, 'pred_label'] = max(cell_pred[idx], 
                                                       key=cell_pred[idx].get)
                present_labels = [k for k, v in cell_pred[idx].items() if v > 0]
                best_label = max(present_labels, key=lambda x: PRIORITY_LOOKUP[x])
                df_cells.loc[idx, 'pred_label_priority'] = best_label
    
    def _draw_cell_rectangles(self, ax, df_selected, color_by='energy', 
                             norm=None, cmap=None, mask_info=None, recon_info=None):
        """Draw cell rectangles with true detector geometry."""
        ax.set_aspect('equal')
        ax.set_xlim(ETA_RANGE)
        ax.set_ylim(PHI_RANGE)
        
        for idx, row in df_selected.iterrows():
            eta_w, phi_w = get_cell_size(row['subCalo'], row['layer'])
            eta0 = row["eta"] - eta_w / 2
            phi0 = row["phi"] - phi_w / 2
            
            if color_by == 'energy':
                energy_val = max(row["energy"], EPS)
                color = cmap(norm(energy_val)) if norm and cmap else 'gray'
                alpha = 0.8
            elif color_by == 'truth':
                color = CLASS_COLORS[int(row["truth_label_priority"])]
                alpha = 0.8
            elif color_by == 'prediction':
                label = int(row["pred_label_priority"]) if row["pred_label_priority"] >= 0 else 0
                color = CLASS_COLORS[label]
                alpha = 0.8
            elif color_by == 'masked':
                is_masked = mask_info['is_masked'][idx] if mask_info is not None else False
                color = 'orange' if is_masked else '#d3d3d3'
                alpha = 0.8 if is_masked else 0.4
            elif color_by == 'reconstruction':
                if recon_info is not None and recon_info['is_masked'][idx]:
                    error = recon_info['errors'][idx]
                    color = cmap(norm(error)) if norm and cmap else 'gray'
                    alpha = 0.8
                else:
                    color = '#e0e0e0'
                    alpha = 0.3
            else:
                color = 'gray'
                alpha = 0.5
            
            rect = Rectangle((eta0, phi0), eta_w, phi_w,
                           facecolor=color, edgecolor='none', 
                           alpha=alpha, zorder=10)
            ax.add_patch(rect)
        
        ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('η')
        ax.set_ylabel('φ')
    
    def plot_energy_panel(self, ax, df_selected, title):
        """Plot energy panel."""
        plot_energy = df_selected['energy'].copy().clip(lower=EPS)
        norm = LogNorm(vmin=EPS, vmax=plot_energy.max())
        cmap = plt.cm.viridis
        
        self._draw_cell_rectangles(ax, df_selected, color_by='energy', 
                                  norm=norm, cmap=cmap)
        ax.set_title(f"{title} Energy", fontsize=14, fontweight='bold')
        return norm, cmap
    
    def plot_truth_panel(self, ax, df_selected, title):
        """Plot truth labels."""
        self._draw_cell_rectangles(ax, df_selected, color_by='truth')
        ax.set_title(f"{title} Truth", fontsize=14, fontweight='bold')
    
    def plot_prediction_panel(self, ax, df_selected, title):
        """Plot prediction labels."""
        self._draw_cell_rectangles(ax, df_selected, color_by='prediction')
        ax.set_title(f"{title} Prediction", fontsize=14, fontweight='bold')
    
    def plot_masked_cells_panel(self, ax, df_selected, mask_info, title):
        """Show which cells were masked."""
        self._draw_cell_rectangles(ax, df_selected, color_by='masked', 
                                  mask_info=mask_info)
        ax.set_title(f"{title} Masked Cells", fontsize=14, fontweight='bold')

    def plot_masked_cells_energy_panel(self, ax, df_selected, mask_info, title):
        """
        Show ONLY masked cells, colored by true energy (log scale) with a
        colorbar — unmasked cells aren't drawn at all, so the color scale
        isn't diluted by the binary masked/unmasked distinction. Answers
        'what real energy activity got hidden', not just 'where'.
        """
        ax.set_aspect('equal')
        ax.set_xlim(ETA_RANGE)
        ax.set_ylim(PHI_RANGE)
    
        if mask_info is None:
            ax.text(0.5, 0.5, "No masking data", ha='center', va='center', fontsize=14)
            ax.set_title(f"{title} Masked Cells (Energy)", fontsize=14, fontweight='bold')
            return
    
        masked_series = mask_info['is_masked']
        masked_idx = [idx for idx in df_selected.index
                      if idx in masked_series.index and bool(masked_series.loc[idx])]
    
        if len(masked_idx) == 0:
            ax.text(0.5, 0.5, "No masked cells in this layer", ha='center', va='center', fontsize=12)
            ax.set_title(f"{title} Masked Cells (Energy)", fontsize=14, fontweight='bold')
            return
    
        df_masked = df_selected.loc[masked_idx]
        plot_energy = df_masked['energy'].copy().clip(lower=EPS)
        norm = LogNorm(vmin=EPS, vmax=max(plot_energy.max(), EPS * 10))
        cmap = plt.cm.viridis
    
        for idx, row in df_masked.iterrows():
            eta_w, phi_w = get_cell_size(row['subCalo'], row['layer'])
            eta0 = row["eta"] - eta_w / 2
            phi0 = row["phi"] - phi_w / 2
            color = cmap(norm(max(row["energy"], EPS)))
            rect = Rectangle((eta0, phi0), eta_w, phi_w,
                              facecolor=color, edgecolor='black', linewidth=0.5,
                              alpha=0.9, zorder=10)
            ax.add_patch(rect)
    
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Energy (log)', fraction=0.046, pad=0.04)
    
        ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('η'); ax.set_ylabel('φ')
        ax.set_title(f"{title} Masked Cells (n={len(masked_idx)})", fontsize=14, fontweight='bold')
    
    def plot_reconstruction_panel(self, ax, df_selected, recon_info, title):
        """Show reconstruction quality."""
        if recon_info is None:
            ax.text(0.5, 0.5, "No reconstruction data", 
                   ha='center', va='center', fontsize=14)
            ax.set_title(f"{title} Reconstruction", fontsize=14, fontweight='bold')
            return
        
        errors = recon_info['errors'].values if hasattr(recon_info['errors'], 'values') else recon_info['errors']
        masked = recon_info['is_masked'].values if hasattr(recon_info['is_masked'], 'values') else recon_info['is_masked']
        
        if masked.any():
            vmax = np.abs(errors[masked]).max() if masked.any() else 1.0
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            cmap = plt.cm.RdBu_r
        else:
            norm = None
            cmap = None
        
        self._draw_cell_rectangles(ax, df_selected, color_by='reconstruction',
                                  norm=norm, cmap=cmap, recon_info=recon_info)
        ax.set_title(f"{title} Reconstruction Error", fontsize=14, fontweight='bold')
    
    def plot_cluster_panel(self, ax, df_cells, cluster_labels, title,
                          cluster_type='truth', mask_info=None):
        """Plot clusters using label-based alignment."""
        ax.set_aspect('equal')
        ax.set_xlim(ETA_RANGE)
        ax.set_ylim(PHI_RANGE)

        if not isinstance(cluster_labels, pd.Series):
            cluster_labels = pd.Series(cluster_labels, index=np.arange(len(cluster_labels)))

        labels_here = cluster_labels.reindex(df_cells.index, fill_value=0)
        unique_clusters = np.unique(labels_here[labels_here > 0])

        if len(unique_clusters) == 0:
            ax.text(0.5, 0.5, f"No {cluster_type} clusters", ha='center', va='center', fontsize=14)
            ax.set_title(f"{title}", fontsize=14, fontweight='bold')
            return

        cluster_cmap = plt.cm.tab20
        norm = plt.Normalize(vmin=1, vmax=max(20, len(unique_clusters)))
        masked_series = mask_info['is_masked'] if mask_info is not None else None

        for idx, row in df_cells.iterrows():
            eta_w, phi_w = get_cell_size(row['subCalo'], row['layer'])
            eta0 = row["eta"] - eta_w / 2
            phi0 = row["phi"] - phi_w / 2

            cluster_id = labels_here.loc[idx]
            color = cluster_cmap(norm((cluster_id - 1) % 20 + 1)) if cluster_id > 0 else '#e0e0e0'
            alpha = 0.7 if cluster_id > 0 else 0.3

            is_masked_cell = bool(masked_series.loc[idx]) if (masked_series is not None and idx in masked_series.index) else False

            rect = Rectangle(
                (eta0, phi0), eta_w, phi_w,
                facecolor=color,
                edgecolor='black' if is_masked_cell else 'none',
                linewidth=1.2 if is_masked_cell else 0,
                alpha=alpha, zorder=10
            )
            ax.add_patch(rect)

        ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('η')
        ax.set_ylabel('φ')
        ax.set_title(f"{title} ({cluster_type.capitalize()} Clusters)", fontsize=14, fontweight='bold')

    def plot_cluster_panel_detailed(self, ax, df_cells, pred_labels_full, truth_labels_full,
                                     match_info, mask_info, title):
        """
        Predicted-cluster panel with enhanced diagnostics:
        - Color: cluster ID
        - Opacity: cell energy (log scale)
        - Black border: masked cell
        - Red hatch: cell belongs to an 'extra' (spurious) predicted cluster
        
        Args:
            ax: matplotlib axis
            df_cells: DataFrame with cell data for this layer
            pred_labels_full: Series with predicted cluster labels for ALL cells
            truth_labels_full: Series with truth cluster labels for ALL cells
            match_info: dict from get_cluster_match_info
            mask_info: dict with 'is_masked' Series
            title: str
        """
        ax.set_aspect('equal')
        ax.set_xlim(ETA_RANGE)
        ax.set_ylim(PHI_RANGE)
        
        # Align labels with current cell selection
        pred_here = pred_labels_full.reindex(df_cells.index, fill_value=0)
        unique_clusters = np.unique(pred_here[pred_here > 0])
        
        if len(unique_clusters) == 0:
            ax.text(0.5, 0.5, "No predicted clusters", ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=14, fontweight='bold')
            return
        
        # Setup colormaps and normalizations
        cluster_cmap = plt.cm.tab20
        norm_cluster = plt.Normalize(vmin=1, vmax=max(20, len(unique_clusters)))
        
        # Energy normalization (log scale)
        max_energy = max(df_cells['energy'].clip(lower=EPS).max(), EPS * 10)
        energy_norm = LogNorm(vmin=EPS, vmax=max_energy)
        
        masked_series = mask_info['is_masked'] if mask_info is not None else None
        
        for idx, row in df_cells.iterrows():
            eta_w, phi_w = get_cell_size(row['subCalo'], row['layer'])
            eta0 = row["eta"] - eta_w / 2
            phi0 = row["phi"] - phi_w / 2
            
            cluster_id = int(pred_here.loc[idx])
            
            # Color by cluster ID, alpha by energy
            if cluster_id > 0:
                color = cluster_cmap(norm_cluster((cluster_id - 1) % 20 + 1))
                alpha = 0.3 + 0.6 * energy_norm(max(row['energy'], EPS))
            else:
                color = '#e0e0e0'
                alpha = 0.2
            
            # Check if masked
            is_masked_cell = (
                masked_series is not None and 
                idx in masked_series.index and 
                bool(masked_series.loc[idx])
            )
            
            # Check if part of extra (unmatched) cluster
            is_extra = (
                cluster_id > 0 and 
                match_info is not None and 
                cluster_id in match_info and 
                not match_info[cluster_id]['matched']
            )
            
            # Set edge properties based on flags
            edgecolor = 'none'
            linewidth = 0
            hatch = None
            
            if is_masked_cell:
                edgecolor = 'black'
                linewidth = 1.2
            
            if is_extra:
                edgecolor = 'red'
                linewidth = 1.5
                hatch = '///'
            
            rect = Rectangle(
                (eta0, phi0), eta_w, phi_w,
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                hatch=hatch,
                alpha=alpha,
                zorder=10
            )
            ax.add_patch(rect)
        
        ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('η')
        ax.set_ylabel('φ')
        ax.set_title(
            f"{title}\n(border=masked, red hatch=extra cluster, opacity=energy)",
            fontsize=11, fontweight='bold'
        )

    def plot_masked_input_panel(self, ax, df_selected, recon_info, title):
        """Shows the model's actual masked input."""
        ax.set_aspect('equal')
        ax.set_xlim(ETA_RANGE)
        ax.set_ylim(PHI_RANGE)

        if recon_info is None:
            ax.text(0.5, 0.5, "No masking data", ha='center', va='center', fontsize=14)
            ax.set_title(f"{title} Masked Input", fontsize=14, fontweight='bold')
            return

        values = recon_info['input_values']
        masked = recon_info['is_masked']
        vmax = np.abs(values).max() if len(values) else 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax) if vmax > 0 else None
        cmap = plt.cm.viridis

        for idx, row in df_selected.iterrows():
            eta_w, phi_w = get_cell_size(row['subCalo'], row['layer'])
            eta0 = row["eta"] - eta_w / 2
            phi0 = row["phi"] - phi_w / 2

            val = values.loc[idx] if idx in values.index else 0.0
            is_masked_cell = bool(masked.loc[idx]) if idx in masked.index else False
            color = cmap(norm(val)) if norm else 'gray'

            rect = Rectangle(
                (eta0, phi0), eta_w, phi_w,
                facecolor=color,
                edgecolor='black' if is_masked_cell else 'none',
                linewidth=1.2 if is_masked_cell else 0,
                alpha=0.9, zorder=10
            )
            ax.add_patch(rect)

        ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('η')
        ax.set_ylabel('φ')
        ax.set_title(f"{title} Masked Input (bordered = masked)", fontsize=14, fontweight='bold')

    def plot_reconstructed_output_panel(self, ax, df_selected, recon_info, title):
        """Shows the model's reconstructed output."""
        ax.set_aspect('equal')
        ax.set_xlim(ETA_RANGE)
        ax.set_ylim(PHI_RANGE)

        if recon_info is None:
            ax.text(0.5, 0.5, "No masking data", ha='center', va='center', fontsize=14)
            ax.set_title(f"{title} Reconstructed Output", fontsize=14, fontweight='bold')
            return

        values = recon_info['recon_values']
        masked = recon_info['is_masked']
        vmax = np.abs(values).max() if len(values) else 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax) if vmax > 0 else None
        cmap = plt.cm.viridis

        for idx, row in df_selected.iterrows():
            eta_w, phi_w = get_cell_size(row['subCalo'], row['layer'])
            eta0 = row["eta"] - eta_w / 2
            phi0 = row["phi"] - phi_w / 2

            val = values.loc[idx] if idx in values.index else 0.0
            is_masked_cell = bool(masked.loc[idx]) if idx in masked.index else False
            color = cmap(norm(val)) if norm else 'gray'

            rect = Rectangle(
                (eta0, phi0), eta_w, phi_w,
                facecolor=color,
                edgecolor='black' if is_masked_cell else 'none',
                linewidth=1.2 if is_masked_cell else 0,
                alpha=0.9, zorder=10
            )
            ax.add_patch(rect)

        ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('η')
        ax.set_ylabel('φ')
        ax.set_title(f"{title} Reconstructed Output (bordered = masked)", fontsize=14, fontweight='bold')

    def generate_event_pdf(self, event_id, model_name, edge_file_path,
                          cluster_validation_results=None, 
                          include_masking=False, cluster_threshold=0.7,
                          event_offset=0, h5f=None, mask_type='random',
                          mask_ratio=0.15, use_not_noise_cut=False):
        """Generate complete PDF for one event with enhanced diagnostics."""
        output_pdf = os.path.join(self.output_dir, 
                                 f"event_{event_id}_{model_name}_composition.pdf")
        
        print(f"  📊 Generating PDF for event {event_id} ({model_name})...")
        df_event, edges_df = self.load_event_data(event_id, edge_file_path, event_offset, h5f)
        
        if df_event.empty:
            print("  ❌ No event data found")
            return None
        
        mask_info = None
        recon_info = None
        if include_masking and self.masking_engine is not None:
            print(f"    🔧 Generating masking data...")
            mask_info, recon_info = self.masking_engine.generate_masking_data(
                df_event, edges_df, event_id, self.h5_path,
                mask_type=mask_type, mask_ratio=mask_ratio, event_offset=event_offset
            )
        
        pred_cluster_labels = None
        truth_cluster_labels = df_event['cluster_index']
        match_info = None
        
        if edges_df is not None and not edges_df.empty:
            validator = ClusterReconstructionValidator()
            n_total_cells = self._get_total_n_cells(h5f)
            final_labels, _ = validator.union_find_clusters(
                edges_df, cluster_threshold, n_cells=n_total_cells,
                use_not_noise_cut=use_not_noise_cut
            )
            pred_cluster_labels = pd.Series(final_labels, index=np.arange(n_total_cells))
            
            # Compute cluster match info for "extra cluster" detection
            match_info = validator.get_cluster_match_info(
                pred_cluster_labels.values, 
                truth_cluster_labels.values
            )
        
        with PdfPages(output_pdf) as pdf:
            for layer_idx, layer_group in enumerate(LAYER_GROUPS):
                if layer_group["layers"] == "all":
                    df_selected = df_event.copy()
                    selected_indices = df_event.index
                else:
                    mask = pd.Series(False, index=df_event.index)
                    for subcalo, layer in layer_group["layers"]:
                        mask |= (df_event["subCalo"] == subcalo) & \
                                (df_event["layer"] == layer)
                    df_selected = df_event[mask]
                    selected_indices = df_selected.index
                
                if include_masking:
                    fig = plt.figure(figsize=(22, 20))
                    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)
                    
                    ax_energy = fig.add_subplot(gs[0, 0])
                    self.plot_energy_panel(ax_energy, df_selected, layer_group['name'])
                    
                    ax_truth = fig.add_subplot(gs[0, 1])
                    self.plot_truth_panel(ax_truth, df_selected, layer_group['name'])
                    
                    ax_pred = fig.add_subplot(gs[0, 2])
                    self.plot_prediction_panel(ax_pred, df_selected, layer_group['name'])
                    
                    sub_mask_info = None
                    if mask_info is not None:
                        sub_mask_info = {
                            'is_masked': mask_info['is_masked'].loc[selected_indices]
                        }
                    
                    sub_recon_info = None
                    if recon_info is not None:
                        sub_recon_info = {
                            'errors': recon_info['errors'].loc[selected_indices],
                            'is_masked': recon_info['is_masked'].loc[selected_indices],
                            'input_values': recon_info['input_values'].loc[selected_indices],
                            'recon_values': recon_info['recon_values'].loc[selected_indices],
                        }
                    
                    ax_masked = fig.add_subplot(gs[1, 0])
                    self.plot_masked_cells_energy_panel(ax_masked, df_selected, 
                                                        sub_mask_info, layer_group['name'])
                    
                    ax_recon = fig.add_subplot(gs[1, 1])
                    self.plot_reconstruction_panel(ax_recon, df_selected, 
                                                  sub_recon_info, layer_group['name'])
                    
                    ax_cluster = fig.add_subplot(gs[1, 2])
                    if pred_cluster_labels is not None:
                        self.plot_cluster_panel_detailed(
                            ax_cluster, df_selected, pred_cluster_labels, 
                            truth_cluster_labels, match_info, sub_mask_info, 
                            layer_group['name']
                        )
                    else:
                        ax_cluster.text(0.5, 0.5, "No predicted clusters",
                                       ha='center', va='center')
                        ax_cluster.set_title(f"{layer_group['name']}", 
                                           fontsize=14, fontweight='bold')
                    
                    ax_masked_input = fig.add_subplot(gs[2, 0])
                    self.plot_masked_input_panel(ax_masked_input, df_selected, 
                                                sub_recon_info, layer_group['name'])
                    
                    ax_recon_output = fig.add_subplot(gs[2, 1])
                    self.plot_reconstructed_output_panel(ax_recon_output, df_selected, 
                                                        sub_recon_info, layer_group['name'])
                else:
                    fig = plt.figure(figsize=(16, 9))
                    gs = fig.add_gridspec(1, 3, hspace=0.2, wspace=0.25)
                    
                    ax_energy = fig.add_subplot(gs[0, 0])
                    self.plot_energy_panel(ax_energy, df_selected, layer_group['name'])
                    
                    ax_truth = fig.add_subplot(gs[0, 1])
                    self.plot_truth_panel(ax_truth, df_selected, layer_group['name'])
                    
                    ax_pred = fig.add_subplot(gs[0, 2])
                    self.plot_prediction_panel(ax_pred, df_selected, layer_group['name'])
                
                fig.suptitle(f"Event {event_id} - {model_name} - {layer_group['name']}",
                           fontsize=16, y=0.98)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Cluster comparison page (enhanced)
            fig_clusters = plt.figure(figsize=(16, 9))
            gs_clusters = fig_clusters.add_gridspec(1, 2, wspace=0.3)
            
            ax_pred_clusters = fig_clusters.add_subplot(gs_clusters[0, 0])
            if pred_cluster_labels is not None:
                self.plot_cluster_panel_detailed(
                    ax_pred_clusters, df_event, pred_cluster_labels, 
                    truth_cluster_labels, match_info, None, "Complete System"
                )
            else:
                ax_pred_clusters.text(0.5, 0.5, "No predicted clusters",
                                     ha='center', va='center', fontsize=14)
            
            ax_truth_clusters = fig_clusters.add_subplot(gs_clusters[0, 1])
            self.plot_cluster_panel(ax_truth_clusters, df_event,
                                   truth_cluster_labels,
                                   "Complete System", 'truth')
            
            fig_clusters.suptitle(f"Event {event_id} - Cluster Comparison\n"
                                 "(red hatch = extra/spurious clusters)",
                                 fontsize=16, y=0.98)
            pdf.savefig(fig_clusters, bbox_inches='tight')
            plt.close()
            
            # Legend page
            fig_legend = plt.figure(figsize=(11, 8.5))
            self.create_legend_page(fig_legend, event_id, model_name, include_masking)
            pdf.savefig(fig_legend, bbox_inches='tight')
            plt.close()
        
        print(f"  ✅ Saved: {os.path.basename(output_pdf)}")
        return output_pdf
    
    def create_legend_page(self, fig, event_id, model_name, include_masking=False):
        """Create comprehensive legend page."""
        fig.suptitle(f"Event {event_id} - {model_name} - Legend", 
                    fontsize=16, y=0.95)
        
        ax1 = fig.add_subplot(311)
        ax1.set_title("Cell Classification Colors", fontsize=14, fontweight='bold')
        legend_patches = [Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) 
                         for i in range(5)]
        ax1.legend(handles=legend_patches, loc='center', ncol=3, fontsize=11)
        ax1.axis('off')
        
        if include_masking:
            ax2 = fig.add_subplot(312)
            ax2.set_title("Masking Visualization", fontsize=14, fontweight='bold')
            mask_patches = [
                Patch(color='orange', label='Masked Cell'),
                Patch(color='#d3d3d3', label='Unmasked Cell'),
                Patch(color='blue', label='Underestimate (Reconstruction)'),
                Patch(color='white', label='Accurate (Reconstruction)'),
                Patch(color='red', label='Overestimate (Reconstruction)'),
            ]
            ax2.legend(handles=mask_patches, loc='center', ncol=3, fontsize=10)
            ax2.axis('off')
        else:
            ax2 = fig.add_subplot(312)
            ax2.text(0.5, 0.5, 
                    "Cluster Style:\nColor = Cluster ID\nSide-by-side comparison available",
                    ha='center', va='center', fontsize=11)
            ax2.axis('off')
        
        ax3 = fig.add_subplot(313)
        info_text = (
            f"• Cell rectangles: True detector cell sizes (varies by layer)\n"
            f"• Energy colors: Log scale (viridis colormap)\n"
            f"• Class colors: See top legend\n"
            f"• All negative energies clipped to 0"
        )
        ax3.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center')
        ax3.axis('off')


# ============================================================================
# COMPONENT 4: IntegratedReportGenerator
# ============================================================================
class IntegratedReportGenerator:
    """Creates single HTML report linking to all artifacts."""
    
    def __init__(self, output_dir, experiment_name):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.folders = create_output_structure(output_dir)
    
    def generate(self, metrics_summary, cluster_validation_summary, 
                event_pdfs, masking_stats=None):
        """Generate HTML report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if metrics_summary:
            df_metrics = pd.DataFrame(metrics_summary)
            df_metrics = df_metrics.sort_values('f1_sum_score', ascending=False)
            metrics_table = df_metrics[['name', 'architecture', 'pretrained', 
                                       'f1_sum_score', 'macro_f1', 'accuracy']
                                     ].head(10).to_html(index=False)
        else:
            metrics_table = "<p>No metrics available</p>"
        
        if cluster_validation_summary is not None and not cluster_validation_summary.empty:
            cluster_table = cluster_validation_summary.head(20).to_html(index=False)
        else:
            cluster_table = "<p>No cluster validation results</p>"
        
        pdf_links = ""
        if event_pdfs:
            for pdf_path in event_pdfs:
                if pdf_path and os.path.exists(pdf_path):
                    rel_path = os.path.relpath(pdf_path, self.folders['reports'])
                    pdf_links += f'<li><a href="{rel_path}">{os.path.basename(pdf_path)}</a></li>\n'
        
        masking_section = ""
        if masking_stats:
            masking_section = f"""
        <div class="container">
            <h2>Masking Analysis</h2>
            <p>Masking visualization panels included in event PDFs.</p>
            <p>Mask type: {masking_stats.get('mask_type', 'N/A')} | 
               Ratio: {masking_stats.get('mask_ratio', 'N/A')}</p>
        </div>
        """
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>CaloGraph Analysis - {self.experiment_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; 
               background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .main-container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: white; text-align: center; font-size: 3em; }}
        .container {{ background-color: white; padding: 30px; border-radius: 15px; 
                     box-shadow: 0 10px 40px rgba(0,0,0,0.2); margin-bottom: 30px; }}
        h2 {{ color: #2c3e50; border-left: 5px solid #3498db; padding-left: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
              color: white; padding: 12px; text-align: center; }}
        td {{ padding: 10px; text-align: center; border-bottom: 1px solid #e0e0e0; }}
        tr:hover {{ background-color: #f5f5f5; }}
        ul {{ list-style-type: none; padding: 0; }}
        li {{ padding: 5px 0; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="main-container">
        <h1>CaloGraph Analysis Report</h1>
        <p style="text-align: center; color: white;">{self.experiment_name} | {timestamp}</p>
        
        <div class="container">
            <h2>Model Performance Summary</h2>
            {metrics_table}
        </div>
        
        {masking_section}
        
        <div class="container">
            <h2>Cluster Reconstruction Validation</h2>
            {cluster_table}
        </div>
        
        <div class="container">
            <h2>Event Visualizations</h2>
            <p>Click to open PDF visualizations:</p>
            <ul>
                {pdf_links if pdf_links else "<li>No event PDFs generated</li>"}
            </ul>
        </div>
        
        <div class="container">
            <h2>Loss Curves</h2>
            <p>Individual model loss curves available in: figures/loss_curves/</p>
        </div>
        
        <div class="container" style="text-align: center; color: #7f8c8d;">
            <p><strong>Class Definitions:</strong></p>
            <p>Class 0: Lone-Lone | Class 1: True-True | Class 2: Cluster-Lone | 
               Class 3: Lone-Cluster | Class 4: Cluster-Cluster</p>
        </div>
    </div>
</body>
</html>"""
        
        report_path = os.path.join(self.folders['reports'], "master_report.html")
        with open(report_path, 'w') as f:
            f.write(html)
        
        print(f"✅ Report saved: {report_path}")
        return report_path


# ============================================================================
# UTILITY: Event Selection
# ============================================================================
def select_events_for_viz(results_df, n_typical=5, n_worst=5, n_best=3, 
                         n_high_multiplicity=2):
    """Stratified event selection."""
    selected_events = []
    
    if results_df is None or results_df.empty:
        return selected_events
    
    typical = results_df.sample(n=min(n_typical, len(results_df)), random_state=42)
    selected_events.extend(typical['event_id'].tolist())
    
    if 'f1_score' in results_df.columns:
        worst = results_df.nsmallest(n_worst, 'f1_score')
        selected_events.extend(worst['event_id'].tolist())
        best = results_df.nlargest(n_best, 'f1_score')
        selected_events.extend(best['event_id'].tolist())
    
    if 'n_truth_clusters' in results_df.columns:
        high_mult = results_df.nlargest(n_high_multiplicity, 'n_truth_clusters')
        selected_events.extend(high_mult['event_id'].tolist())
    
    seen = set()
    unique_events = []
    for e in selected_events:
        if e not in seen:
            seen.add(e)
            unique_events.append(e)
    
    return unique_events


# ============================================================================
# MAIN ORCHESTRATION (DEBUG PROCESSES ALL PHASES WITH 1 EVENT)
# ============================================================================
def main():
    args = parse_args()
    
    print("="*80)
    print("🎯 CaloGraph Unified Analysis & Visualization Suite")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.debug:
        print("🔧 DEBUG MODE: Processing ALL phases with only 1 event")
    
    cluster_thresholds = [float(t) for t in args.cluster_thresholds.split(',')]
    event_selection = {}
    for part in args.events_to_visualize.split(','):
        if ':' in part:
            key, val = part.split(':')
            event_selection[key.strip()] = int(val.strip())
    
    folders = create_output_structure(args.output_dir)
    
    pkl_files = sorted(glob.glob(os.path.join(args.models_dir, "*_metrics.pkl")))
    pkl_files = [f for f in pkl_files if 'pretrain_metrics' not in os.path.basename(f)]
    model_names = [os.path.basename(f).replace("_metrics.pkl", "") for f in pkl_files]
    
    print(f"\n📊 Found {len(model_names)} models")
    
    if not model_names:
        print("❌ No models found!")
        return
    
    h5_path = os.path.join(args.data_dir, args.h5_file)
    
    # Get event IDs first (needed for debug mode)
    all_event_ids = set()
    for model_name in model_names:
        parquet_path = os.path.join(args.parquet_dir, f"results_{model_name}.parquet")
        if os.path.exists(parquet_path):
            events = pq.read_table(parquet_path, columns=['event_id']).to_pandas()
            all_event_ids.update(events['event_id'].unique()[:100])
    
    if args.debug:
        event_ids_sorted = sorted(all_event_ids)[:1]
        debug_event_id = event_ids_sorted[0] if event_ids_sorted else args.event_offset
        print(f"  🔧 DEBUG: Using event {debug_event_id}")
    elif args.max_events_cluster is not None:
        event_ids_sorted = sorted(all_event_ids)[:args.max_events_cluster]
        debug_event_id = None
    else:
        event_ids_sorted = sorted(all_event_ids)
        debug_event_id = None
    
    models_to_process = model_names
    
    # ========================================================================
    # PHASE 1: Model Metrics (RUNS in debug with 1 event only)
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: Model Metrics Analysis (incl. Loss Curves)")
    print("="*80)
    if args.debug:
        print("  🔧 DEBUG: Using only 1 event for metrics")
    
    metrics_analyzer = ModelMetricsAnalyzer(
        args.models_dir,
        args.parquet_dir, 
        args.output_dir,
        max_rows_roc=args.max_rows_roc,
        max_rows_confusion=args.max_rows_confusion,
        batch_size=args.batch_size
    )
    all_results = metrics_analyzer.generate_all_metrics(
        models_to_process, debug=args.debug, debug_event_id=debug_event_id
    )
    
    # ========================================================================
    # PHASE 2: Cluster Validation
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Cluster Reconstruction Validation")
    print("="*80)
    
    cluster_validator = ClusterReconstructionValidator(
        confidence_thresholds=cluster_thresholds
    )
    
    # Open HDF5 once for Phase 2 and Phase 3
    with h5py.File(h5_path, 'r') as h5f:
        cluster_results = {}
        delta_r_results_all = {}

        # Phase 2b — CC-confusion ablation
        if args.include_cc_ablation:
            print("\n" + "="*80)
            print("PHASE 2b: CC-Confusion Ablation (Options A & B)")
            print("="*80)
            for model_name in models_to_process:
                print(f"\n  Running ablation for: {model_name}")
                save_path = os.path.join(folders['cluster_ablation'],
                                        f"ablation_{model_name}.parquet")
                ablation_df = cluster_validator.validate_all_events_ablation(
                    event_ids_sorted, args.parquet_dir, model_name, h5f, save_path,
                    event_offset=args.event_offset,
                    cc_heavy_threshold=args.cc_heavy_threshold
                )
                cluster_validator.plot_ablation_comparison(
                    ablation_df, folders['cluster_ablation'], model_name
                )
                print(f"  ✅ Ablation results: {len(ablation_df)} rows")

        for model_name in models_to_process:
            print(f"\n  Validating clusters for: {model_name}")
            save_path = os.path.join(folders['cluster_validation'], 
                                    f"cluster_validation_{model_name}.parquet")
            results_df = cluster_validator.validate_all_events(
                event_ids_sorted, args.parquet_dir, model_name, h5f, save_path,
                event_offset=args.event_offset,
                use_not_noise_cut=args.use_not_noise_cut,
                output_folder=folders['iou_distributions']
            )
            cluster_results[model_name] = results_df
            print(f"  ✅ Results: {len(results_df)} rows")
            
            # Generate efficiency maps, energy metrics, and ΔR matching
            print(f"\n  Generating efficiency maps, energy metrics, and ΔR matching...")
            delta_r_summaries = []
            
            for event_id in event_ids_sorted:
                edges_df = cluster_validator.load_event_edges(event_id, args.parquet_dir, model_name)
                truth_labels = cluster_validator.load_truth_clusters(event_id, h5f, args.event_offset)
                
                if edges_df is None or truth_labels is None:
                    continue
                
                h5_event_id = event_id - args.event_offset
                cell_eta = h5f['cell/cell_eta'][h5_event_id]
                cell_phi = h5f['cell/cell_phi'][h5_event_id]
                cell_energy = h5f['cell/cell_e'][h5_event_id]
                
                # Use visualization threshold for clustering
                pred_labels, _ = cluster_validator.union_find_clusters(
                    edges_df, args.visualization_threshold, n_cells=len(truth_labels),
                    use_not_noise_cut=args.use_not_noise_cut
                )
                
                # Efficiency map (model-based)
                cluster_validator.plot_efficiency_eta_phi(
                    pred_labels, truth_labels, cell_eta, cell_phi,
                    folders['efficiency_maps'], model_name, event_id
                )
                
                # TRUTH CHECK efficiency map (structural limitations)
                cluster_validator.plot_efficiency_eta_phi_truth_check(
                    edges_df, truth_labels, cell_eta, cell_phi,
                    len(truth_labels), folders['efficiency_maps'], 
                    model_name, event_id
                )

                # Energy metrics
                energy_metrics = cluster_validator.compute_energy_weighted_metrics(
                    pred_labels, truth_labels, cell_energy
                )
                
                if energy_metrics:
                    energy_df = pd.DataFrame(energy_metrics)
                    energy_save_path = os.path.join(
                        folders['energy_metrics'], f"energy_{model_name}_event{event_id}.parquet"
                    )
                    energy_df.to_parquet(energy_save_path)
                    
                    # Generate visualization automatically
                    cluster_validator.plot_energy_metrics_visualization(
                        energy_df, folders['energy_metrics'], model_name, event_id
                    )
                
                # ΔR-based matching
                delta_r_results, delta_r_summary = cluster_validator.match_clusters_by_delta_r(
                    pred_labels, truth_labels, cell_eta, cell_phi,
                    cell_energy, args.delta_r
                )
                
                delta_r_summaries.append({
                    'event_id': event_id,
                    **delta_r_summary
                })
                
                # Generate per-event ΔR details for first 5 events
                if len(delta_r_summaries) <= 5:
                    cluster_validator.plot_delta_r_event_details(
                        delta_r_results, event_id, folders['delta_r_matching'], model_name
                    )
            
            # Save and visualize ΔR matching results for this model
            if delta_r_summaries:
                delta_r_df = pd.DataFrame(delta_r_summaries)
                delta_r_save_path = os.path.join(
                    folders['cluster_validation'], 
                    f"delta_r_matching_{model_name}.parquet"
                )
                delta_r_df.to_parquet(delta_r_save_path)
                delta_r_results_all[model_name] = delta_r_df
                
                # Generate summary visualization
                cluster_validator.plot_delta_r_matching_summary(
                    delta_r_df, folders['delta_r_matching'], model_name
                )
                
                print(f"  ✅ ΔR matching results: {len(delta_r_df)} events")
            
            print(f"  ✅ Efficiency maps, energy metrics, and ΔR matching saved")
        
        # ====================================================================
        # PHASE 3: Event Selection
        # ====================================================================
        if args.debug:
            events_to_viz = [debug_event_id] if debug_event_id else [args.event_offset]
            print(f"\n  🔧 DEBUG: Using event {events_to_viz[0]}")
        else:
            print("\n" + "="*80)
            print("PHASE 3: Event Selection")
            print("="*80)
            
            first_model = model_names[0]
            parquet_path = os.path.join(args.parquet_dir, f"results_{first_model}.parquet")
            
            if os.path.exists(parquet_path):
                events_df = pq.read_table(
                    parquet_path,
                    columns=['event_id', 'true_label', 'pred_label', 'confidence']
                ).to_pandas()
                
                per_event_metrics = []
                for event_id, group in events_df.groupby('event_id'):
                    if len(group) > 0:
                        correct = (group['true_label'] == group['pred_label']).sum()
                        accuracy = correct / len(group)
                        
                        truth_labels = cluster_validator.load_truth_clusters(
                            event_id, h5f, args.event_offset
                        )
                        n_truth_clusters = (
                            len(np.unique(truth_labels[truth_labels > 0]))
                            if truth_labels is not None else 0
                        )
                        
                        per_event_metrics.append({
                            'event_id': event_id,
                            'f1_score': accuracy,
                            'n_edges': len(group),
                            'n_truth_clusters': n_truth_clusters
                        })
                
                per_event_df = pd.DataFrame(per_event_metrics)
                events_to_viz = select_events_for_viz(
                    per_event_df,
                    n_typical=event_selection.get('typical', 5),
                    n_worst=event_selection.get('worst', 5),
                    n_best=event_selection.get('best', 3),
                    n_high_multiplicity=event_selection.get('high_multiplicity', 2)
                )
                print(f"  Selected {len(events_to_viz)} events")
            else:
                print("  ⚠️ No parquet file found, using defaults")
                events_to_viz = [args.event_offset, args.event_offset + 1, args.event_offset + 2]
    
    # ========================================================================
    # PHASE 4a: Masking engine setup
    # ========================================================================
    auto_checkpoints = {}
    if args.auto_masking_viz:
        print("\n" + "="*80)
        print("PHASE 4a: Auto-Discovering Pretrained Checkpoints")
        print("="*80)
        auto_checkpoints = discover_pretrained_checkpoints(args.models_dir)
        print(f"  🔍 Found {len(auto_checkpoints)} pretrained checkpoint(s):")
        for name, info in auto_checkpoints.items():
            print(f"     • {name} -> mask_type={info['mask_type']}, "
                  f"ckpt={os.path.basename(info['checkpoint'])}")

    masking_engine_cache = {}

    def get_masking_engine_for_model(model_name):
        """Return (engine, mask_type) for this model, or (None, None)."""
        if args.auto_masking_viz:
            info = auto_checkpoints.get(model_name)
            if info is None or info['mask_type'] is None:
                return None, None
            ckpt = info['checkpoint']
            if ckpt not in masking_engine_cache:
                print(f"  🔧 Loading masking engine for {model_name} ({info['mask_type']})...")
                masking_engine_cache[ckpt] = MaskingInferenceEngine(
                    model_path=ckpt, model_type=args.model_type,
                    hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                    num_heads=args.num_heads
                )
            return masking_engine_cache[ckpt], info['mask_type']
        elif args.include_masking_viz and args.pretrained_model:
            ckpt = args.pretrained_model
            if ckpt not in masking_engine_cache:
                print("\n" + "="*80)
                print("PHASE 4a: Initializing Masking Inference Engine")
                print("="*80)
                masking_engine_cache[ckpt] = MaskingInferenceEngine(
                    model_path=ckpt, model_type=args.model_type,
                    hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                    num_heads=args.num_heads
                )
            return masking_engine_cache[ckpt], args.mask_type
        return None, None
    
    # ========================================================================
    # PHASE 4: Event Visualization
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: Event Visualization")
    print("="*80)
    
    event_visualizer = EventVisualizer(folders['event_pdfs'], h5_path)
    
    if args.debug:
        models_to_viz = model_names
    elif args.max_models_viz is not None:
        models_to_viz = model_names[:args.max_models_viz]
    else:
        models_to_viz = model_names
    
    pdf_paths = []
    any_masking_used = False
    
    with h5py.File(h5_path, 'r') as h5f:
        for model_name in models_to_viz:
            engine, mask_type_for_model = get_masking_engine_for_model(model_name)
            event_visualizer.set_masking_engine(engine)
            include_masking_for_model = engine is not None
            if include_masking_for_model:
                any_masking_used = True

            edge_file_path = os.path.join(args.parquet_dir, f"results_{model_name}.parquet")
            
            for event_id in events_to_viz:
                cluster_val = None
                if model_name in cluster_results:
                    event_results = cluster_results[model_name]
                    if 'event_id' in event_results.columns:
                        event_results = event_results[event_results['event_id'] == event_id]
                        if not event_results.empty:
                            cluster_val = event_results.iloc[0].to_dict()
                
                pdf_path = event_visualizer.generate_event_pdf(
                    event_id, model_name, edge_file_path,
                    cluster_validation_results=cluster_val,
                    include_masking=include_masking_for_model,
                    cluster_threshold=args.visualization_threshold,
                    event_offset=args.event_offset,
                    h5f=h5f,
                    mask_type=mask_type_for_model or args.mask_type,
                    mask_ratio=args.mask_ratio,
                    use_not_noise_cut=args.use_not_noise_cut
                )
                pdf_paths.append(pdf_path)
    
    # ========================================================================
    # PHASE 5: Generate Report (RUNS in debug too)
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 5: Integrated Report")
    print("="*80)
    
    report_generator = IntegratedReportGenerator(
        args.output_dir,
        f"Analysis_{datetime.now().strftime('%Y%m%d')}"
    )
    
    summary_cluster = cluster_results.get(model_names[0], pd.DataFrame())
    
    masking_stats = None
    if any_masking_used:
        masking_stats = {
            'mask_type': args.mask_type if not args.auto_masking_viz else 'auto (per-model)',
            'mask_ratio': args.mask_ratio
        }
    
    report_path = report_generator.generate(
        all_results, summary_cluster, pdf_paths, masking_stats
    )
    print(f"\n📄 Report: {report_path}")
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\n📊 Results saved to: {args.output_dir}")
    print(f"📁 Event PDFs: {folders['event_pdfs']}")
    print(f"🔬 Cluster validation: {folders['cluster_validation']}")
    print(f"📊 IoU distributions: {folders['iou_distributions']}")
    print(f"🗺️ Efficiency maps: {folders['efficiency_maps']}")
    print(f"⚡ Energy metrics: {folders['energy_metrics']}")
    print(f"📈 Loss curves: {folders['loss_curves']}")
    print(f"🎯 ΔR matching: {folders['delta_r_matching']}")
    print(f"📄 Report: {report_path}")
    
    if args.include_masking_viz or args.auto_masking_viz:
        print(f"🎭 Masking visualization: {'ENABLED' if any_masking_used else 'DISABLED (no matching checkpoint found)'}")


if __name__ == "__main__":
    main()