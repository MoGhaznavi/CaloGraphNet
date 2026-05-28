#!/usr/bin/env python3
"""
Graph Foundation Model for Particle Physics Edge Classification

FULLY INTEGRATED & OPTIMIZED VERSION with:
- Multiple architectures (GCN, GAT, Graph Transformer, GraphSAGE)
- Intelligent loss functions (CrossEntropy, Focal, Weighted)
- Feature selection (--baseline: 3 features, --all-features: 42+ features)
- Proper file outputs (PKL for metrics, Parquet for predictions)
- VECTORIZED OPERATIONS for 3-5x faster training
- Single GPU or multi-GPU training

Usage examples:
    # Train GCN baseline
    python3 new_RunningMultipleModels.py --model gcn --baseline --gpu 0
    
    # Train Graph Transformer with all features
    python3 new_RunningMultipleModels.py --model transformer --all-features --gpu 1
    
    # Train GAT with focal loss
    python3 new_RunningMultipleModels.py --model gat --weighted-loss --weight-strategy focal --gpu 2
    
    # Train all architectures sequentially
    python3 new_RunningMultipleModels.py --model all --epochs 30 --gpu 0
    
    # Convert old pickle results
    python3 new_RunningMultipleModels.py convert path/to/results.pkl
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard Library
import argparse
import datetime
import gc
import glob
import json
import os
import pickle
import re
import signal
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Libraries
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

# PyTorch Geometric
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.nn import BatchNorm, LayerNorm
from torch.nn import BatchNorm1d, LayerNorm as LayerNorm1d

# Parquet support
import pyarrow as pa
import pyarrow.parquet as pq

# HDF5 support
import h5py

try:
    from debug_utils import (
        DEBUG_CONFIG, debug_print, tensor_stats, check_gradients,
        inspect_model_weights, debug_forward_pass, debug_loss_and_backward,
        debug_batch_accuracy, deep_dive_single_batch
    )
    DEBUG_AVAILABLE = True
except ImportError:
    DEBUG_AVAILABLE = False
    def debug_print(*args, **kwargs): pass

# ============================================================================
# GLOBAL SETUP
# ============================================================================

# Set multiprocessing start method to 'spawn' (required for CUDA on some systems)
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

# Enable cuDNN auto-tuning for faster GPU operations
torch.backends.cudnn.benchmark = True

# Limit CPU threads (slightly increased for better performance)
torch.set_num_threads(4)


# ============================================================================
# ARGPARSE CONFIGURATION
# ============================================================================

def parse_args():
    """Parse command line arguments for flexible model configuration."""
    parser = argparse.ArgumentParser(
        description='Graph Foundation Model for Particle Physics Edge Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model architecture
    parser.add_argument('--model', '-m', type=str, default='gcn',
                        choices=['gcn', 'gat', 'transformer', 'sage', 'all'],
                        help='Backbone architecture. Use "all" to train all architectures sequentially')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--layers', '-l', type=int, default=6,
                        help='Number of backbone layers')
    parser.add_argument('--heads', type=int, default=2,
                        help='Number of attention heads (for GAT and Transformer)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--layer-weights', action='store_true',
                        help='Use learnable layer weights')
    parser.add_argument('--softmax-weights', action='store_true',
                        help='Apply softmax to layer weights')
    
    # Normalization
    parser.add_argument('--norm', type=str, default='batch',
                        choices=['batch', 'layer', 'none'],
                        help='Normalization type')
    
    # Feature configuration
    parser.add_argument('--baseline', action='store_true', default=True,
                        help='Use baseline features only (SNR, eta, phi) - 3 features')
    parser.add_argument('--all-features', action='store_true',
                        help='Use all 42+ features from dataset')
    
    # Training configuration
    parser.add_argument('--epochs', '-e', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='Batch size (events per batch)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Loss weighting
    parser.add_argument('--weighted-loss', action='store_true',
                        help='Use weighted loss for class imbalance')
    parser.add_argument('--weight-strategy', type=str, default='inverse',
                        choices=['inverse', 'focal', 'logarithmic', 'manual'],
                        help='Class weighting strategy')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Alpha parameter for focal loss')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    
    # Data split
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Train/test split ratio')
    
    # Hardware
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='Enable mixed precision training (default: True)')
    parser.add_argument('--no-mixed-precision', action='store_false', dest='mixed_precision',
                        help='Disable mixed precision training')

    # Experiment tracking
    parser.add_argument('--save-dir', type=str, default='/storage/mxg1065/foundation_experiments',
                        help='Directory to save models')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--data-dir', type=str, default='/storage/mxg1065/datafiles',
                        help='Data directory')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume from checkpoint if available')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (only runs a few events)')
    parser.add_argument('--inference-only', action='store_true',  # NEW FLAG
                        help='Skip training, only run inference on best model and save results') 
    return parser.parse_args()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log(msg: str) -> None:
    """Simple timestamped logger."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def save_pickle(data: Any, filepath: str) -> None:
    """Save data to pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath: str) -> Any:
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def find_latest_checkpoint(save_dir: str, base_name: str) -> Optional[Tuple[int, str]]:
    """Find the newest model checkpoint in a directory."""
    pattern = re.compile(f"{re.escape(os.path.splitext(base_name)[0])}_epoch(\\d+).pt")
    checkpoints = []
    
    for fname in os.listdir(save_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, os.path.join(save_dir, fname)))
    
    return max(checkpoints, key=lambda x: x[0]) if checkpoints else None


# ============================================================================
# FEATURE LOADING WITH SELECTION (BASELINE vs ALL FEATURES)
# ============================================================================

def load_features_with_selection(data_dir: str, args) -> Tuple:
    """
    Load features based on --baseline or --all-features flag.
    
    Returns:
        features_dict: Dict mapping event_id -> feature array (num_cells, num_features)
        unscaled_dict: Dict mapping event_id -> unscaled feature array
        pairs: Edge connectivity array (num_edges, 2)
        labels: Edge labels array (num_events, num_edges)
        cluster_info: Dict mapping event_id -> cluster information
        input_dim: Number of input features
        feature_names: List of feature names
    """
    log(f"📊 Loading features with mode: {'BASELINE (3 features)' if args.baseline else 'ALL FEATURES (42+)'}")
    
    # Load static cell features
    cells_path = os.path.join(data_dir, "cells.npy")
    cells = np.load(cells_path)
    num_cells = cells.shape[0]
    log(f"  ✓ Loaded {num_cells} cells from cells.npy")
    
    # Load edge connectivity
    pairs_path = os.path.join(data_dir, "pairs.npy")
    pairs = np.load(pairs_path).astype(np.int32)
    num_edges = pairs.shape[0]
    log(f"  ✓ Loaded {num_edges} edges from pairs.npy")
    
    # Load events HDF5
    events_path = os.path.join(data_dir, "events.h5")
    log(f"  Loading HDF5: {events_path}")
    
    with h5py.File(events_path, 'r') as h5f:
        # Determine num_events from dataset shape (attribute not present in this HDF5)
        if 'cell/snr_computed' in h5f:
            num_events = h5f['cell/snr_computed'].shape[0]
        elif 'cell/energy_raw' in h5f:
            num_events = h5f['cell/energy_raw'].shape[0]
        elif 'cell/snr_raw' in h5f:
            num_events = h5f['cell/snr_raw'].shape[0]
        else:
            num_events = 1000  # fallback
        log(f"  Found {num_events} events")
            
        if args.baseline:
            # ================================================================
            # BASELINE MODE: Only SNR, eta, phi (3 features)
            # ================================================================
            
            # Load SNR
            if 'cell/snr_computed' in h5f:
                snr = h5f['cell/snr_computed'][:]
                log("  ✓ Using computed SNR (energy/noise)")
            elif 'cell/snr_raw' in h5f:
                snr = h5f['cell/snr_raw'][:]
                log("  ✓ Using raw SNR from ROOT")
            elif 'cell/energy_raw' in h5f and 'cell/noise_raw' in h5f:
                energy = h5f['cell/energy_raw'][:]
                noise = h5f['cell/noise_raw'][:]
                noise = np.where(noise == 0, 1e-6, noise)
                snr = energy / noise
                log("  ✓ Computed SNR from energy/noise")
            else:
                raise ValueError("No SNR or energy/noise data found in events.h5")
            
            # Load geometry
            if 'cell/cell_eta' in h5f and 'cell/cell_phi' in h5f:
                eta = h5f['cell/cell_eta'][:]
                phi = h5f['cell/cell_phi'][:]
                log("  ✓ Loaded eta/phi from events.h5")
            else:
                # Use static geometry from cells.npy
                eta = np.tile(cells['eta_event0'].astype(np.float32), (num_events, 1))
                phi = np.tile(cells['phi_event0'].astype(np.float32), (num_events, 1))
                log("  ⚠️  Using eta/phi from cells.npy")
            
            # Build feature dict: 3 features per cell
            features_dict = {}
            unscaled_dict = {}
            
            for ev in range(num_events):
                features = np.stack([
                    snr[ev],
                    eta[ev],
                    phi[ev]
                ], axis=1).astype(np.float32)
                
                features_dict[ev] = features
                unscaled_dict[ev] = features.copy()  # Baseline uses same for scaled/unscaled
                
                if args.debug and ev % 100 == 0 and ev > 0:
                    log(f"    Processed {ev}/{num_events} events")
            
            input_dim = 3
            feature_names = ['snr', 'eta', 'phi']
            
        else:
            # ================================================================
            # ALL FEATURES MODE: Comprehensive feature set (42+ features)
            # ================================================================
            
            feature_list = []
            feature_names = []
            
            # ----------------------------------------------------------------
            # 1. ENERGY/SNR FEATURES (2-3 features)
            # ----------------------------------------------------------------
            
            # SNR (computed or raw)
            if 'cell/snr_computed' in h5f:
                snr = h5f['cell/snr_computed'][:]
                feature_list.append(snr)
                feature_names.append('snr_computed')
                log("  ✓ Added: snr_computed")
            elif 'cell/snr_raw' in h5f:
                snr = h5f['cell/snr_raw'][:]
                feature_list.append(snr)
                feature_names.append('snr_raw')
                log("  ✓ Added: snr_raw")
            
            # Raw energy
            if 'cell/energy_raw' in h5f:
                energy = h5f['cell/energy_raw'][:]
                feature_list.append(energy)
                feature_names.append('energy_raw')
                log("  ✓ Added: energy_raw")
            
            # Raw noise
            if 'cell/noise_raw' in h5f:
                noise_raw = h5f['cell/noise_raw'][:]
                feature_list.append(noise_raw)
                feature_names.append('noise_raw')
                log("  ✓ Added: noise_raw")
            
            # ----------------------------------------------------------------
            # 2. GEOMETRY FEATURES (2-5 features)
            # ----------------------------------------------------------------
            
            if 'cell/cell_eta' in h5f and 'cell/cell_phi' in h5f:
                eta = h5f['cell/cell_eta'][:]
                phi = h5f['cell/cell_phi'][:]
            else:
                eta = np.tile(cells['eta_event0'].astype(np.float32), (num_events, 1))
                phi = np.tile(cells['phi_event0'].astype(np.float32), (num_events, 1))
            
            feature_list.extend([eta, phi])
            feature_names.extend(['eta', 'phi'])
            log("  ✓ Added: eta, phi")
            
            # Geometry derivatives from cells.npy
            if 'deta' in cells.dtype.names:
                deta = np.tile(cells['deta'].astype(np.float32), (num_events, 1))
                feature_list.append(deta)
                feature_names.append('deta')
                log("  ✓ Added: deta")
            
            if 'dphi' in cells.dtype.names:
                dphi = np.tile(cells['dphi'].astype(np.float32), (num_events, 1))
                feature_list.append(dphi)
                feature_names.append('dphi')
                log("  ✓ Added: dphi")
            
            if 'volume' in cells.dtype.names:
                volume = np.tile(cells['volume'].astype(np.float32), (num_events, 1))
                feature_list.append(volume)
                feature_names.append('volume')
                log("  ✓ Added: volume")
            
            # ----------------------------------------------------------------
            # 3. NOISE STATISTICS (4 features) - VERY VALUABLE!
            # ----------------------------------------------------------------
            
            if 'noise_mean' in cells.dtype.names:
                noise_mean = np.tile(cells['noise_mean'].astype(np.float32), (num_events, 1))
                feature_list.append(noise_mean)
                feature_names.append('noise_mean')
                log("  ✓ Added: noise_mean")
            
            if 'noise_std' in cells.dtype.names:
                noise_std = np.tile(cells['noise_std'].astype(np.float32), (num_events, 1))
                feature_list.append(noise_std)
                feature_names.append('noise_std')
                log("  ✓ Added: noise_std")
            
            if 'noise_count' in cells.dtype.names:
                noise_count = np.tile(cells['noise_count'].astype(np.float32), (num_events, 1))
                feature_list.append(noise_count)
                feature_names.append('noise_count')
                log("  ✓ Added: noise_count")
            
            if 'noise_category' in cells.dtype.names:
                noise_category = np.tile(cells['noise_category'].astype(np.float32), (num_events, 1))
                feature_list.append(noise_category)
                feature_names.append('noise_category')
                log("  ✓ Added: noise_category")
            
            # ----------------------------------------------------------------
            # 4. TOPOLOGY FEATURES
            # ----------------------------------------------------------------
            
            if 'num_neighbors' in cells.dtype.names:
                num_neighbors = np.tile(cells['num_neighbors'].astype(np.float32), (num_events, 1))
                feature_list.append(num_neighbors)
                feature_names.append('num_neighbors')
                log("  ✓ Added: num_neighbors")
            
            # ----------------------------------------------------------------
            # 5. DETECTOR INFO (one-hot encoded)
            # ----------------------------------------------------------------
            
            if 'subcalo' in cells.dtype.names:
                subcalo = cells['subcalo'].astype(np.int32)
                # One-hot encode (assuming 0-3 values)
                num_subcalo = max(4, subcalo.max() + 1)
                for i in range(num_subcalo):
                    onehot = np.tile((subcalo == i).astype(np.float32), (num_events, 1))
                    feature_list.append(onehot)
                    feature_names.append(f'subcalo_{i}')
                log(f"  ✓ Added: subcalo one-hot ({num_subcalo} classes)")
            
            if 'sampling' in cells.dtype.names:
                sampling = cells['sampling'].astype(np.int32)
                # One-hot encode (assuming 0-2 values)
                num_sampling = max(3, sampling.max() + 1)
                for i in range(num_sampling):
                    onehot = np.tile((sampling == i).astype(np.float32), (num_events, 1))
                    feature_list.append(onehot)
                    feature_names.append(f'sampling_{i}')
                log(f"  ✓ Added: sampling one-hot ({num_sampling} classes)")
            
            # ----------------------------------------------------------------
            # 6. CLUSTER FEATURES
            # ----------------------------------------------------------------
            
            if 'cell/cell_cluster_index' in h5f:
                cluster_idx = h5f['cell/cell_cluster_index'][:]
                
                # Cluster membership flag
                in_cluster = (cluster_idx >= 0).astype(np.float32)
                feature_list.append(in_cluster)
                feature_names.append('in_cluster')
                log("  ✓ Added: in_cluster")
                
                # Cluster ID (normalized)
                max_cluster = max(1, cluster_idx.max())
                cluster_id_norm = cluster_idx.astype(np.float32) / max_cluster
                feature_list.append(cluster_id_norm)
                feature_names.append('cluster_id_norm')
                log("  ✓ Added: cluster_id_norm")
            
            # ----------------------------------------------------------------
            # 7. ADDITIONAL BRANCHES from events.h5
            # ----------------------------------------------------------------
            
            if 'cell' in h5f:
                cell_group = h5f['cell']
                skip_keys = {
                    'cell_SNR_scaled', 'cell_SNR_raw', 'snr_computed', 'snr_raw',
                    'cell_eta', 'cell_phi', 'energy_raw', 'noise_raw',
                    'cell_cluster_index', 'cell_to_cluster_eta', 'cell_to_cluster_phi'
                }
                
                for key in cell_group.keys():
                    if key in skip_keys:
                        continue
                    
                    try:
                        data = cell_group[key][:]
                        if data.ndim == 2 and data.shape[0] == num_events:
                            feature_list.append(data.astype(np.float32))
                            feature_names.append(key)
                            log(f"  ✓ Added: {key}")
                    except Exception as e:
                        log(f"  ⚠️  Skipped {key}: {str(e)[:50]}")
            
            # Build feature dict
            features_dict = {}
            unscaled_dict = {}
            
            for ev in range(num_events):
                event_features = []
                for feat in feature_list:
                    if feat.ndim == 2 and feat.shape[0] == num_events:
                        event_features.append(feat[ev])
                    else:
                        event_features.append(feat)
                
                features = np.stack(event_features, axis=1).astype(np.float32)
                features_dict[ev] = features
                unscaled_dict[ev] = features.copy()
                
                if args.debug and ev % 100 == 0 and ev > 0:
                    log(f"    Processed {ev}/{num_events} events")
            
            input_dim = len(feature_names)
        
        # Load cluster info
        cluster_info = {}
        if 'cell/cell_cluster_index' in h5f:
            cluster_idx = h5f['cell/cell_cluster_index'][:]
            for ev in range(num_events):
                cluster_info[ev] = {'cell_cluster_index': cluster_idx[ev].astype(np.int32)}
        
        # Load labels
        labels_path = os.path.join(data_dir, "labels.npy")
        labels = np.load(labels_path).astype(np.int8)
        log(f"  ✓ Loaded labels: {labels.shape}")
    
    # Verify
    log("\n✅ DATASET VERIFICATION:")
    log(f"   Events: {num_events}")
    log(f"   Cells: {num_cells}")
    log(f"   Edges: {num_edges}")
    log(f"   Labels shape: {labels.shape}")
    log(f"   Feature shape per event: {list(features_dict[0].shape)}")
    if len(feature_names) > 10:
        log(f"   Feature names ({len(feature_names)}): {feature_names[:10]}...")
    else:
        log(f"   Feature names: {feature_names}")
    
    return features_dict, unscaled_dict, pairs, labels, cluster_info, input_dim, feature_names


# ============================================================================
# LOSS FUNCTIONS (OPTIMIZED WITH GPU COMPUTATION)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance.
    Supports class-specific alpha values for better minority class focus.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Can be:
            - float: Single alpha for all classes
            - list/tensor: Class-specific alpha values [α_0, α_1, ..., α_C]
        gamma: Focusing parameter (higher = more focus on hard examples)
        weight: Optional per-class weights (computed separately)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha: Union[float, List[float], torch.Tensor] = 0.25, 
                 gamma: float = 2.0,
                 weight: Optional[torch.Tensor] = None, 
                 reduction: str = 'mean',
                 chunk_size: int = 100000):  # Process in chunks of 100k edges
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.chunk_size = chunk_size
        
        # Handle different alpha types
        if isinstance(alpha, (float, int)):
            self.alpha = float(alpha)
            self.per_class_alpha = None
        else:
            if isinstance(alpha, list):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.per_class_alpha = alpha
            self.alpha = None
            
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Memory-efficient forward pass using chunking.
        """
        num_samples = inputs.size(0)
        total_loss = 0.0
        num_chunks = 0
        
        # Process in chunks to save memory
        for start_idx in range(0, num_samples, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_samples)
            
            inputs_chunk = inputs[start_idx:end_idx]
            targets_chunk = targets[start_idx:end_idx]
            
            # Compute cross-entropy loss for chunk
            ce_loss = F.cross_entropy(
                inputs_chunk, targets_chunk, 
                weight=self.weight, reduction='none'
            )
            
            # Compute p_t = exp(-CE_loss)
            pt = torch.exp(-ce_loss)
            
            # Get alpha_t for each sample in chunk
            if self.per_class_alpha is not None:
                alpha_t = self.per_class_alpha.to(inputs.device)[targets_chunk]
            else:
                alpha_t = self.alpha
            
            # Compute Focal Loss for chunk
            focal_loss_chunk = alpha_t * (1 - pt) ** self.gamma * ce_loss
            
            total_loss += focal_loss_chunk.sum()
            num_chunks += 1
            
            # Free memory
            del inputs_chunk, targets_chunk, ce_loss, pt, focal_loss_chunk
        
        # Apply reduction
        if self.reduction == 'mean':
            return total_loss / num_samples
        elif self.reduction == 'sum':
            return total_loss
        else:
            # For 'none', we'd need to return all, but that uses memory
            # Return mean as fallback
            return total_loss / num_samples

def compute_class_weights(labels: np.ndarray, num_classes: int, 
                          strategy: str = 'inverse', device: torch.device = None,
                          **kwargs) -> torch.Tensor:
    """
    Compute class weights for imbalanced classification.
    OPTIMIZED: Uses PyTorch on GPU for faster computation.
    """
    # Move to GPU for faster computation if available
    if device is not None and device.type == 'cuda':
        labels_tensor = torch.as_tensor(labels, device=device).flatten()
        counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    else:
        counts_np = np.bincount(labels.flatten(), minlength=num_classes)
        counts = torch.tensor(counts_np, dtype=torch.float32)
    
    log(f"Class counts: {dict(zip(range(num_classes), counts.int().tolist()))}")
    
    if strategy == 'focal':
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        total = len(labels.flatten())
        class_weights = (total - counts) / (counts + 1e-5)
        weights = alpha * class_weights ** gamma
        
    elif strategy == 'logarithmic':
        weights = 1.0 / torch.log1p(counts + 1e-5)
        
    elif strategy == 'manual':
        # Strategic weights: prioritize rare but important classes
        manual = torch.tensor([0.1, 10.0, 8.0, 8.0, 15.0], device=counts.device)
        weights = manual[:num_classes]
        
    else:  # 'inverse'
        weights = 1.0 / (counts + 1e-5)
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    
    log(f"Computed weights ({strategy}): {weights.tolist()}")
    
    return weights.to(device) if device is not None else weights


def create_loss_function(args, labels: np.ndarray, device: torch.device) -> nn.Module:
    """
    Create appropriate loss function based on arguments.
    """
    num_classes = 5
    
    # RECOMMENDED CLASS-SPECIFIC ALPHAS FOR YOUR DATASET
    RECOMMENDED_ALPHAS = [0.10, 0.60, 0.70, 0.70, 1.00]  # Classes 0,1,2,3,4
    
    if args.weighted_loss:
        class_weights = compute_class_weights(
            labels, num_classes, 
            strategy=args.weight_strategy,
            device=device,
            alpha=args.focal_alpha,
            gamma=args.focal_gamma
        )
        
        if args.weight_strategy == 'focal':
            # Use class-specific alphas for extreme imbalance
            loss_fn = FocalLoss(
                alpha=RECOMMENDED_ALPHAS,  # Class-specific!
                gamma=args.focal_gamma,    # Still controllable via --focal-gamma
                weight=None
            )
            log(f"✅ Using Focal Loss with class-specific alphas: {RECOMMENDED_ALPHAS}")
            log(f"   (No additional class weights - using alphas only)")
            log(f"   Gamma: {args.focal_gamma}")
        else:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            log(f"✅ Using Weighted CrossEntropyLoss ({args.weight_strategy})")
    else:
        loss_fn = nn.CrossEntropyLoss()
        log("✅ Using standard CrossEntropyLoss")
    
    return loss_fn

# ============================================================================
# MODEL ARCHITECTURES (OPTIMIZED FORWARD PASS)
# ============================================================================

class GraphFoundationModel(nn.Module):
    """
    Unified Graph Neural Network supporting multiple architectures.
    STRICT EXACT REPLICA of old MultiEdgeClassifier for GCN.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 device: torch.device,
                 model_type: str = 'gcn',
                 num_layers: int = 6,
                 num_heads: int = 2,
                 dropout: float = 0.0,
                 layer_weights: bool = False,
                 softmax_weights: bool = False,
                 norm_type: str = 'batch',
                 debug: bool = False):
        super().__init__()
        
        self.device = device
        self.model_type = model_type
        self.debug = debug
        self.num_layers = num_layers
        self.layer_weights_enabled = layer_weights  # Boolean flag like old code
        self.softmax = softmax_weights              # Named 'softmax' like old code
        
        # ============ EXACT OLD CODE INITIALIZATION ORDER ============
        # 1. Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # 2. Convolution layers (named 'convs')
        self.convs = nn.ModuleList()
        if model_type == 'gcn':
            for _ in range(num_layers):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif model_type == 'gat':
            for _ in range(num_layers):
                self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        elif model_type == 'transformer':
            for _ in range(num_layers):
                self.convs.append(TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        elif model_type == 'sage':
            for _ in range(num_layers):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 3. BatchNorm layers (named 'bns' EXACTLY like old code)
        if norm_type == 'batch':
            self.bns = nn.ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        elif norm_type == 'layer':
            self.bns = nn.ModuleList([LayerNorm1d(hidden_dim) for _ in range(num_layers)])
        elif norm_type == 'none':
            self.bns = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        
        # 4. Edge classifier (named 'fc' EXACTLY like old code)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        
        # 5. Layer weights - EXACT old code behavior
        if layer_weights:
            self.layer_weights = nn.Parameter(torch.ones(num_layers))
        # CRITICAL: When layer_weights=False, NO attribute is created (not even None)
        
        # ============ NO EXTRA ATTRIBUTES ============
        # No self.norms alias
        # No self.edge_classifier alias  
        # No self.dropout_layer (nn.Identity or otherwise)
        # No self._attention_weights
        # No self.softmax_weights (use self.softmax)
        # No register_buffer calls
        # =============================================
        
        log(f"📐 Initialized {model_type.upper()} model (STRICT OLD CODE REPLICA):")
        log(f"   Input dim: {input_dim}, Hidden dim: {hidden_dim}")
        log(f"   Layers: {num_layers}, Norm: {norm_type}")
        log(f"   Edge classifier: fc (Linear 2*{hidden_dim} -> {output_dim})")
        log(f"   Layer weights: {'Enabled' if layer_weights else 'Disabled'}")
    
    def forward(self,
                x_list: List[torch.Tensor],
                edge_index_list: List[torch.Tensor],
                edge_index_out_list: List[torch.Tensor],
                y_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass - EXACT MATCH to old MultiEdgeClassifier.
        """
        all_edge_reprs = []
        
        # Optional layer-weight normalization
        # Uses boolean flag like old code
        if self.layer_weights_enabled:
            weights = (torch.softmax(self.layer_weights, dim=0)
                       if self.softmax else self.layer_weights)
        else:
            weights = None
        
        # Process each graph in the batch
        for x, proc_edges, orig_edges in zip(x_list, edge_index_list, edge_index_out_list):
            x = x.to(self.device, non_blocking=True)
            proc_edges = proc_edges.to(self.device, non_blocking=True)
            
            # Initial embedding (NO dropout, NO activation)
            x_embed = self.node_embedding(x)
            
            # Apply GCN layers with residual connections
            for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
                # Message passing + normalization + activation
                h = torch.relu(bn(conv(x_embed, proc_edges)))
                
                # Optional layer weighting
                if weights is not None:
                    h = weights[i] * h
                
                # Residual connection (NO dropout)
                x_embed = x_embed + h
            
            # Build edge-level representations
            src, dst = orig_edges[0], orig_edges[1]
            edge_repr = torch.cat([x_embed[src], x_embed[dst]], dim=-1)
            all_edge_reprs.append(edge_repr)
        
        # Final classification
        out = self.fc(torch.cat(all_edge_reprs, dim=0))
        
        # DEBUG
        if self.debug:
            print(f"[DEBUG] Model output dtype: {out.dtype}")
            print(f"[DEBUG] Model output requires grad: {out.requires_grad}")
        
        return out

# ============================================================================
# DATA GENERATOR (OPTIMIZED WITH PINNED MEMORY)
# ============================================================================

class MultiClassBatchGenerator(IterableDataset):
    """
    Optimized IterableDataset for large events.
    OPTIMIZED: Uses pinned memory for faster GPU transfers.
    """
    
    def __init__(
        self,
        features_dict: Dict[int, np.ndarray],
        neighbor_pairs: np.ndarray,
        labels: np.ndarray,
        mode: str = "train",
        is_bi_directional: bool = True,
        batch_size: int = 1,
        train_ratio: float = 0.7,
        debug: bool = False,
        unscaled_data_dict: Optional[Dict[int, np.ndarray]] = None,
        cluster_info_dict: Optional[Dict[int, Dict]] = None,
    ):
        self.debug = debug
        self.is_bi_directional = is_bi_directional
        self.batch_size = batch_size
        self.cluster_info_dict = cluster_info_dict
        self.num_events = len(features_dict)
        self.num_pairs = neighbor_pairs.shape[0]
        
        # Validate labels
        if labels.ndim != 2:
            raise ValueError(f"Labels must be 2D array, got shape {labels.shape}")
        
        # OPTIMIZED: Pre-convert to tensors with pinned memory for faster GPU transfer
        self.features_dict = {}
        for k, v in features_dict.items():
            t = torch.as_tensor(v, dtype=torch.float32)
            if torch.cuda.is_available():
                t = t.pin_memory()
            self.features_dict[k] = t
        
        if unscaled_data_dict is not None:
            self.unscaled_features_dict = {}
            for k, v in unscaled_data_dict.items():
                t = torch.as_tensor(v, dtype=torch.float32)
                if torch.cuda.is_available():
                    t = t.pin_memory()
                self.unscaled_features_dict[k] = t
        else:
            self.unscaled_features_dict = None
        
        # Pin memory for pairs and labels
        self.neighbor_pairs = torch.as_tensor(neighbor_pairs, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        if torch.cuda.is_available():
            self.neighbor_pairs = self.neighbor_pairs.pin_memory()
            self.labels = self.labels.pin_memory()
        
        # Split events
        split_idx = int(self.num_events * train_ratio)
        
        if mode == "train":
            self.event_indices = list(range(0, split_idx))
        else:
            self.event_indices = list(range(split_idx, self.num_events))
        
        log(f"📊 {mode.upper()} SET: Events {self.event_indices[0]}-{self.event_indices[-1]} "
            f"({len(self.event_indices)} events)")
        
        # Precompute samples
        self.precomputed_samples = self._precompute_all_samples()
    
    def _precompute_all_samples(self) -> List[Tuple]:
        """Precompute all event samples for fast iteration."""
        samples: List[Tuple] = []
        
        pairs_t = self.neighbor_pairs.T
        
        for event_idx in self.event_indices:
            if event_idx not in self.features_dict:
                continue
            
            x_scaled = self.features_dict[event_idx]
            x_unscaled = (self.unscaled_features_dict.get(event_idx)
                          if self.unscaled_features_dict else None)
            
            out_labels = self.labels[event_idx]
            out_labels = out_labels.unsqueeze(1) if out_labels.dim() == 1 else out_labels
            
            cluster_info = (self.cluster_info_dict.get(event_idx) 
                           if self.cluster_info_dict else None)
            
            samples.append((x_scaled, pairs_t, pairs_t.clone(), 
                           out_labels, x_unscaled, cluster_info))
        
        return samples
    
    def __iter__(self):
        for s in self.precomputed_samples:
            yield s
    
    def __len__(self):
        return len(self.precomputed_samples)
    
    @staticmethod
    def collate_data(batch: List[Tuple]) -> Tuple:
        """Combine list of samples into batch format."""
        x_list = [b[0] for b in batch]
        edge_index_list = [b[1] for b in batch]
        edge_index_out_list = [b[2] for b in batch]
        y_batch = torch.cat([b[3] for b in batch], dim=0)
        unscaled_list = None if batch[0][4] is None else [b[4] for b in batch]
        cluster_info_list = None if batch[0][5] is None else [b[5] for b in batch]
        return x_list, edge_index_list, edge_index_out_list, y_batch, unscaled_list, cluster_info_list


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, scaler, device: torch.device,
                debug: bool = False, accumulation_steps: int = 1, 
                epoch: int = 0) -> Dict[str, float]:
    """Train the model for one epoch with optional debugging."""
    model.train()
    
    total_loss = 0.0
    correct = total = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    # Determine if we're using mixed precision
    use_amp = scaler is not None
    
    for batch_idx, batch in enumerate(loader):
        
        # ============ DEBUGGING INTEGRATION ============
        if DEBUG_AVAILABLE and DEBUG_CONFIG['enabled']:
            # Deep dive on first batch of first epoch
            if epoch == 0 and batch_idx == 0 and debug:
                debug_print("\n🔬 PERFORMING DEEP DIVE ON FIRST BATCH", force=True)
                deep_dive_single_batch(model, batch, criterion, optimizer, device)
            
            # Debug forward pass
            if batch_idx < 3 or debug:  # Debug first 3 batches
                logits, y_batch, _ = debug_forward_pass(
                    model, batch, epoch * len(loader) + batch_idx, device
                )
                
                # Compute loss with debugging
                loss_val = debug_loss_and_backward(
                    logits, y_batch, criterion, model, optimizer, scaler,
                    epoch * len(loader) + batch_idx
                )
                
                if loss_val is None:  # NaN or Inf loss
                    debug_print("❌ Aborting batch due to invalid loss", force=True)
                    continue
                
                loss = torch.tensor(loss_val, device=device) / accumulation_steps
                
                # Debug accuracy
                acc = debug_batch_accuracy(logits, y_batch, epoch * len(loader) + batch_idx)
                
                # Handle optimizer step for debug path
                if (batch_idx + 1) % accumulation_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                
                # Track metrics
                total_loss += loss_val * len(y_batch)
                correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total += len(y_batch)
                
                continue  # Skip normal training flow for debugged batches
        # ============ END DEBUGGING ============
        
        # Normal training flow
        x_list, edge_idx_list, edge_idx_out_list, y_batch, _, _ = batch
        
        # Move to device
        x_list = [x.to(device, non_blocking=True) for x in x_list]
        edge_idx_list = [e.to(device, non_blocking=True) for e in edge_idx_list]
        edge_idx_out_list = [e.to(device, non_blocking=True) for e in edge_idx_out_list]
        y_batch = y_batch.to(device, non_blocking=True).squeeze(1)
        
        # Forward pass - NO AUTOCAST (we handle precision manually)
        scores = model(x_list, edge_idx_list, edge_idx_out_list)
        loss = criterion(scores, y_batch) / accumulation_steps
        
        # Force float32 for loss to prevent gradient underflow
        loss = loss.float()
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                # Unscale gradients before clipping (optional but recommended)
                scaler.unscale_(optimizer)
                # Optional: Gradient clipping for stability
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Optional: Gradient clipping for stability
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Track metrics
        total_loss += loss.item() * len(y_batch) * accumulation_steps
        preds = scores.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    
    # Handle remaining gradients
    if total > 0 and (batch_idx + 1) % accumulation_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    
    return {
        "loss": total_loss / total if total else 0,
        "acc": correct / total if total else 0,
    }

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, use_amp: bool = False, num_classes: int = 5) -> Dict[str, float]:
    """Evaluate the model with comprehensive metrics: Recall, Precision, F1 (Macro + Weighted)."""
    model.eval()
    total_loss = correct = total = 0
    
    # Full confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for x_list, edge_idx_list, edge_idx_out_list, y_batch, _, _ in loader:
            x_list = [x.to(device, non_blocking=True) for x in x_list]
            edge_idx_list = [e.to(device, non_blocking=True) for e in edge_idx_list]
            edge_idx_out_list = [e.to(device, non_blocking=True) for e in edge_idx_out_list]
            y_batch = y_batch.to(device, non_blocking=True).squeeze(1)
            
            # Forward pass
            scores = model(x_list, edge_idx_list, edge_idx_out_list)
            loss = criterion(scores, y_batch)
            
            # Force float32
            loss = loss.float()
            
            total_loss += loss.item() * len(y_batch)
            preds = scores.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
            
            # Build confusion matrix
            y_np = y_batch.cpu().numpy()
            preds_np = preds.cpu().numpy()
            for i in range(len(y_np)):
                cm[y_np[i], preds_np[i]] += 1
    
    # ===== Compute all metrics from confusion matrix =====
    
    # Per-class components
    TP = np.diag(cm)                           # True Positives (diagonal elements)
    FP = cm.sum(axis=0) - TP                   # False Positives (column sum minus diagonal)
    FN = cm.sum(axis=1) - TP                   # False Negatives (row sum minus diagonal)
    class_totals = cm.sum(axis=1)              # Total actual samples per class
    class_weights = class_totals / cm.sum()    # Class frequency weights
    
    # Per-class metrics
    recall = np.zeros(num_classes)      # True Positive Rate / Sensitivity
    precision = np.zeros(num_classes)   # Positive Predictive Value
    f1 = np.zeros(num_classes)          # Harmonic mean of Precision and Recall
    
    for c in range(num_classes):
        recall[c] = TP[c] / (TP[c] + FN[c]) if (TP[c] + FN[c]) > 0 else 0.0
        precision[c] = TP[c] / (TP[c] + FP[c]) if (TP[c] + FP[c]) > 0 else 0.0
        f1[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0.0
    
    # ===== Macro metrics (unweighted average - each class counts equally) =====
    macro_recall = np.mean(recall)
    macro_precision = np.mean(precision)
    macro_f1 = np.mean(f1)
    
    # ===== Weighted metrics (weighted by class frequency) =====
    weighted_recall = np.sum(class_weights * recall)
    weighted_precision = np.sum(class_weights * precision)
    weighted_f1 = np.sum(class_weights * f1)
    
    # ===== Sum Scores (Normalized: 1.0 = random guessing, num_classes = perfect) =====
    recall_sum_score = macro_recall * num_classes       # RSS: Recall Sum Score
    f1_sum_score = macro_f1 * num_classes               # FSS: F1 Sum Score
    weighted_recall_score = weighted_recall * num_classes
    weighted_f1_score = weighted_f1 * num_classes       # WFSS: Weighted F1 Sum Score
    
    # ===== Build results dictionary =====
    results = {
        # Loss & basic accuracy
        "loss": total_loss / total if total else 0,
        "accuracy": correct / total if total else 0,
        
        # Per-class metrics
        **{f"recall_class_{c}": recall[c] for c in range(num_classes)},
        **{f"precision_class_{c}": precision[c] for c in range(num_classes)},
        **{f"f1_class_{c}": f1[c] for c in range(num_classes)},
        **{f"randomness_class_{c}": recall[c] * num_classes for c in range(num_classes)},
        
        # Macro (unweighted) metrics - each class counts equally (for physics!)
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        
        # Weighted metrics - weighted by class frequency
        "weighted_recall": weighted_recall,
        "weighted_precision": weighted_precision,
        "weighted_f1": weighted_f1,
        
        # Sum Scores (baseline = 1.0, perfect = num_classes)
        "randomness_metric": recall_sum_score,          # RSS (original - Recall only)
        "f1_sum_score": f1_sum_score,                   # FSS (NEW - FP-aware!)
        "weighted_recall_score": weighted_recall_score,
        "weighted_f1_score": weighted_f1_score,          # WFSS (NEW)
        
        # Raw confusion data for later analysis
        "class_totals": class_totals.tolist(),
        "confusion_matrix": cm.tolist(),
    }
    
    return results

@torch.no_grad()
def run_inference(model: nn.Module, generator: MultiClassBatchGenerator, 
                  device: torch.device, debug: bool = False, show_progress: bool = True) -> List[Dict]:
    """Run inference event-by-event with optional progress bar."""
    model.eval()
    all_results = []
    
    total_events = len(generator)
    
    # Setup progress tracking
    if show_progress and not debug:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(generator), total=total_events, 
                          desc="   🔮 Inference", unit="events", 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            log("   (Install tqdm for progress bar: pip install tqdm)")
    else:
        use_tqdm = False
    
    # Use appropriate iterator
    if use_tqdm:
        iterator = iterator
    else:
        iterator = enumerate(generator)
    
    for i, (x_scaled, edge_index, edge_index_out, y, x_unscaled, cluster_info) in iterator:
        if hasattr(generator, 'event_indices') and i < len(generator.event_indices):
            event_id = generator.event_indices[i]
        else:
            event_id = i
        
        # Simple progress update if not using tqdm
        if show_progress and not use_tqdm and not debug and (i + 1) % 50 == 0:
            pct = (i + 1) / total_events * 100
            elapsed = time.perf_counter() - t0 if 't0' in locals() else 0
            log(f"      Progress: {i + 1}/{total_events} events ({pct:.1f}%)")
        
        if i == 0:
            t0 = time.perf_counter()
        
        x_scaled = x_scaled.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        edge_index_out = edge_index_out.to(device, non_blocking=True)
        
        out = model([x_scaled], [edge_index], [edge_index_out])
        preds = out.argmax(dim=1).cpu().numpy()
        scores = torch.softmax(out, dim=1).cpu().numpy()
        
        src_nodes = edge_index_out[0].cpu().numpy()
        dst_nodes = edge_index_out[1].cpu().numpy()
        
        labels_np = None
        if y is not None:
            if y.dim() == 2 and y.shape[1] == 1:
                labels_np = y.squeeze(1).numpy()
            else:
                labels_np = y.numpy()
        
        all_results.append({
            "event_id": event_id,
            "preds": preds,
            "scores": scores,
            "labels": labels_np,
            "neighbor_pairs": np.stack([src_nodes, dst_nodes], axis=1),
            "cluster_info": cluster_info,
        })
        
        if debug and i >= 4:
            log("Debug mode: stopping after 5 events")
            break
    
    # Final summary
    if show_progress and not debug and total_events > 0:
        elapsed = time.perf_counter() - t0
        log(f"   ✅ Inference complete: {len(all_results)} events in {elapsed:.1f}s ({elapsed/len(all_results):.2f}s/event)")
    
    return all_results

def save_results_to_parquet(results: List[Dict], save_path: str, 
                            model_name: str, cluster_info_dict: Dict = None) -> str:
    """
    Save inference results to Parquet format - OPTIMIZED VERSION.
    Uses chunking, better compression, and auto-deletes existing files.
    """
    if not results:
        return ""
    
    # ============ DELETE EXISTING FILE TO AVOID SCHEMA MISMATCH ============
    if os.path.exists(save_path):
        os.remove(save_path)
        log(f"      Removed existing file: {os.path.basename(save_path)}")
    # =====================================================================

    log("   💾 Saving to Parquet (optimized)...")
    total_edges = sum(len(r['preds']) for r in results)
    log(f"      Total edges: {total_edges:,}")

    # Process in chunks to reduce memory
    chunk_size = 10  # Events per chunk
    writer = None
    
    # Determine if cluster info is available (check first result)
    has_cluster_info = False
    if results and results[0].get('cluster_info') is not None:
        cluster_idx = results[0]['cluster_info'].get('cell_cluster_index')
        has_cluster_info = (cluster_idx is not None)
    
    if has_cluster_info:
        log(f"      Including cluster information in output")

    for chunk_start in range(0, len(results), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(results))
        chunk_results = results[chunk_start:chunk_end]

        # Calculate chunk size
        chunk_edges = sum(len(r['preds']) for r in chunk_results)

        # Pre-allocate arrays for this chunk
        event_ids = np.zeros(chunk_edges, dtype=np.int32)
        edge_ids = np.zeros(chunk_edges, dtype=np.int32)
        source_ids = np.zeros(chunk_edges, dtype=np.int32)
        target_ids = np.zeros(chunk_edges, dtype=np.int32)
        true_labels = np.full(chunk_edges, -1, dtype=np.int8)
        pred_labels = np.zeros(chunk_edges, dtype=np.int8)
        confidence = np.zeros(chunk_edges, dtype=np.float32)
        scores = np.zeros((chunk_edges, 5), dtype=np.float32)
        
        # Pre-allocate cluster arrays if needed
        if has_cluster_info:
            source_cluster = np.full(chunk_edges, -1, dtype=np.int32)
            target_cluster = np.full(chunk_edges, -1, dtype=np.int32)

        # Fill arrays
        offset = 0
        for result in chunk_results:
            n = len(result['preds'])
            
            event_ids[offset:offset+n] = result['event_id']
            edge_ids[offset:offset+n] = np.arange(n)
            source_ids[offset:offset+n] = result['neighbor_pairs'][:, 0]
            target_ids[offset:offset+n] = result['neighbor_pairs'][:, 1]
            
            if result['labels'] is not None:
                true_labels[offset:offset+n] = result['labels']
            
            pred_labels[offset:offset+n] = result['preds']
            confidence[offset:offset+n] = result['scores'][np.arange(n), result['preds']]
            scores[offset:offset+n] = result['scores']
            
            # Add cluster info if available
            if has_cluster_info and result.get('cluster_info') is not None:
                cidx = result['cluster_info'].get('cell_cluster_index', None)
                if cidx is not None:
                    source_cluster[offset:offset+n] = cidx[result['neighbor_pairs'][:, 0]]
                    target_cluster[offset:offset+n] = cidx[result['neighbor_pairs'][:, 1]]
            
            offset += n
        
        # Create DataFrame for this chunk
        df_chunk = pd.DataFrame({
            'event_id': event_ids,
            'edge_id': edge_ids,
            'source_id': source_ids,
            'target_id': target_ids,
            'true_label': true_labels,
            'pred_label': pred_labels,
            'confidence': confidence,
            'score_class_0': scores[:, 0],
            'score_class_1': scores[:, 1],
            'score_class_2': scores[:, 2],
            'score_class_3': scores[:, 3],
            'score_class_4': scores[:, 4],
            'model_name': model_name,
        })
        
        # Add cluster columns if available
        if has_cluster_info:
            df_chunk['source_cluster'] = source_cluster
            df_chunk['target_cluster'] = target_cluster
            df_chunk['same_cluster'] = (df_chunk['source_cluster'] == df_chunk['target_cluster']) & (df_chunk['source_cluster'] >= 0)

        # Write chunk
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(
                save_path,
                table.schema,
                compression='zstd',  # Better compression than snappy
                compression_level=3,  # Balance speed vs size
                use_dictionary=True,
                write_statistics=True,
            )

        writer.write_table(table)

        # Progress update
        pct = (chunk_end) / len(results) * 100
        log(f"      Progress: {chunk_end}/{len(results)} events ({pct:.1f}%)")

        # Free memory
        del df_chunk, table, event_ids, edge_ids, source_ids, target_ids
        del true_labels, pred_labels, confidence, scores
        if has_cluster_info:
            del source_cluster, target_cluster
        gc.collect()

    if writer is not None:
        writer.close()

    # Get file size
    file_size = os.path.getsize(save_path) / (1024**3)
    log(f"   💾 Results saved to: {save_path} ({file_size:.2f} GB)")
    return save_path

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model_full(model: nn.Module, 
                     train_loader: DataLoader,
                     test_loader: DataLoader,
                     test_generator: MultiClassBatchGenerator,
                     optimizer: optim.Optimizer,
                     criterion: nn.Module,
                     scaler,
                     device: torch.device,
                     args,
                     model_name: str,
                     cluster_info: Dict = None) -> Dict:
    """Complete training loop with checkpointing and comprehensive evaluation."""
    
    # ============ DEBUGGING: Initial model inspection ============
    if DEBUG_AVAILABLE and DEBUG_CONFIG['enabled']:
        debug_print("\n" + "="*80, force=True)
        debug_print("🔍 INITIAL MODEL INSPECTION", force=True)
        debug_print("="*80, force=True)
        
        zero_weights = inspect_model_weights(model, detailed=True)
        
        if zero_weights:
            debug_print("\n⚠️  WARNING: Model has zero-initialized weights!", force=True)
        else:
            debug_print("\n✅ All weights properly initialized", force=True)
        
        debug_print("\n📊 TRAINING DATA CLASS DISTRIBUTION:", force=True)
        sample_batch = next(iter(train_loader))
        _, _, _, y_batch, _, _ = sample_batch
        debug_print(f"  Sample batch labels: {torch.unique(y_batch, return_counts=True)}", force=True)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    model_path = os.path.join(args.save_dir, model_name)
    best_model_path = os.path.join(args.save_dir, f"best_{model_name}")
    metrics_path = os.path.splitext(model_path)[0] + "_metrics.pkl"
    
    best_epoch = 0
    best_f1_sum_score = 0.0  # Use F1 Sum Score for model selection (FP-aware!)
    best_randomness_metric = 0.0
    best_macro_f1 = 0.0
    start_epoch = 1
    
    num_classes = 5
    
    # Metrics tracking - comprehensive (NO MCC)
    metrics = {
        "train_loss": [], "test_loss": [],
        "train_accuracy": [], "test_accuracy": [],
        "test_macro_recall": [], "test_macro_precision": [], "test_macro_f1": [],
        "test_weighted_recall": [], "test_weighted_precision": [], "test_weighted_f1": [],
        "test_randomness_metric": [], "test_f1_sum_score": [],
        "test_weighted_recall_score": [], "test_weighted_f1_score": [],
        "test_recall_class_0": [], "test_recall_class_1": [], "test_recall_class_2": [], 
        "test_recall_class_3": [], "test_recall_class_4": [],
        "test_precision_class_0": [], "test_precision_class_1": [], "test_precision_class_2": [],
        "test_precision_class_3": [], "test_precision_class_4": [],
        "test_f1_class_0": [], "test_f1_class_1": [], "test_f1_class_2": [],
        "test_f1_class_3": [], "test_f1_class_4": [],
        "epoch_times": [],
        "best_accuracy": 0.0,
        "best_macro_recall": 0.0,
        "best_macro_precision": 0.0,
        "best_macro_f1": 0.0,
        "best_weighted_recall": 0.0,
        "best_weighted_precision": 0.0,
        "best_weighted_f1": 0.0,
        "best_randomness_metric": 0.0,
        "best_f1_sum_score": 0.0,
        "best_weighted_recall_score": 0.0,
        "best_weighted_f1_score": 0.0,
        "best_epoch": 0,
        "total_time": 0.0,
        "args": vars(args),
    }
    
    # Resume from checkpoint
    if args.resume:
        chk = find_latest_checkpoint(args.save_dir, model_name)
        if chk:
            resumed_epoch, chk_path = chk
            checkpoint = torch.load(chk_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = resumed_epoch + 1
            log(f"[Resume] Loaded checkpoint: {chk_path}")
            
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'rb') as f:
                        saved_metrics = pickle.load(f)
                    metrics.update(saved_metrics)
                    best_f1_sum_score = metrics.get("best_f1_sum_score", 0.0)
                    best_randomness_metric = metrics.get("best_randomness_metric", 0.0)
                    best_epoch = metrics.get("best_epoch", 0)
                except:
                    pass
    
    early_counter = 0
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        
        train_res = train_epoch(model, train_loader, optimizer, criterion, scaler, device, args.debug, epoch=epoch)
        test_res = evaluate(model, test_loader, criterion, device, use_amp=(scaler is not None))
        
        dt = time.perf_counter() - t0
        
        # Store all metrics
        metrics["epoch_times"].append(dt)
        metrics["train_loss"].append(train_res["loss"])
        metrics["train_accuracy"].append(train_res["acc"])
        metrics["test_loss"].append(test_res["loss"])
        metrics["test_accuracy"].append(test_res["accuracy"])
        metrics["test_macro_recall"].append(test_res["macro_recall"])
        metrics["test_macro_precision"].append(test_res["macro_precision"])
        metrics["test_macro_f1"].append(test_res["macro_f1"])
        metrics["test_weighted_recall"].append(test_res["weighted_recall"])
        metrics["test_weighted_precision"].append(test_res["weighted_precision"])
        metrics["test_weighted_f1"].append(test_res["weighted_f1"])
        metrics["test_randomness_metric"].append(test_res["randomness_metric"])
        metrics["test_f1_sum_score"].append(test_res["f1_sum_score"])
        metrics["test_weighted_recall_score"].append(test_res["weighted_recall_score"])
        metrics["test_weighted_f1_score"].append(test_res["weighted_f1_score"])
        
        for c in range(num_classes):
            metrics[f"test_recall_class_{c}"].append(test_res.get(f'recall_class_{c}', 0.0))
            metrics[f"test_precision_class_{c}"].append(test_res.get(f'precision_class_{c}', 0.0))
            metrics[f"test_f1_class_{c}"].append(test_res.get(f'f1_class_{c}', 0.0))
        
        # ============ Best model tracking using F1 SUM SCORE (FP-aware!) ============
        if test_res["f1_sum_score"] > best_f1_sum_score + 0.01:
            best_f1_sum_score = test_res["f1_sum_score"]
            best_randomness_metric = test_res["randomness_metric"]
            best_macro_f1 = test_res["macro_f1"]
            best_epoch = epoch
            
            # Update all best metrics
            for key in ["accuracy", "macro_recall", "macro_precision", "macro_f1",
                       "weighted_recall", "weighted_precision", "weighted_f1",
                       "randomness_metric", "f1_sum_score", "weighted_recall_score", "weighted_f1_score"]:
                metrics[f"best_{key}"] = test_res[key]
            metrics["best_epoch"] = best_epoch
            
            early_counter = 0
            
            if not args.debug:
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                if scaler is not None:
                    checkpoint_dict["scaler_state_dict"] = scaler.state_dict()
                torch.save(checkpoint_dict, best_model_path)
                log(f"  💾 Best model saved (F1_Sum={best_f1_sum_score:.2f}, Macro_F1={best_macro_f1:.4f}, Recall_Sum={best_randomness_metric:.2f})")
        else:
            early_counter += 1
        
        # Save checkpoint
        if not args.debug:
            chk_path = os.path.join(args.save_dir, f"{os.path.splitext(model_name)[0]}_epoch{epoch}.pt")
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler_state_dict"] = scaler.state_dict()
            torch.save(checkpoint_dict, chk_path)
        
        if args.debug or epoch % 5 == 0:
            log(f"[Epoch {epoch}] Time {dt:.1f}s  "
                f"F1_Sum={test_res['f1_sum_score']:.2f}  Recall_Sum={test_res['randomness_metric']:.2f}  "
                f"Macro_F1={test_res['macro_f1']:.4f}  Acc={test_res['accuracy']:.4f}  "
                f"Best_F1_Sum={best_f1_sum_score:.2f}")
        
        if early_counter >= args.patience:
            log(f"[Early Stop] Triggered at epoch {epoch}")
            break
    
    metrics["total_time"] = sum(metrics["epoch_times"])
    metrics["time_per_epoch"] = np.mean(metrics["epoch_times"]) if metrics["epoch_times"] else 0
    
    # Final inference with best model
    if not args.debug and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        final_results = run_inference(model, test_generator, device, args.debug)
        metrics["num_events_evaluated"] = len(final_results)
        
        model_base = os.path.splitext(model_name)[0]
        parquet_path = os.path.join(args.save_dir, f"results_{model_base}.parquet")
        save_results_to_parquet(final_results, parquet_path, model_base, cluster_info)
        metrics["parquet_results_path"] = parquet_path
        
        save_pickle(metrics, metrics_path)
        log(f"📊 Metrics saved to: {metrics_path}")
    
    return metrics, model, best_model_path

# ============================================================================
# SINGLE MODEL TRAINING
# ============================================================================

def train_single_model(args, model_type: str = None) -> Tuple[Dict, str]:
    """Train a single model with the given configuration."""
    
    # Determine model type
    if model_type is not None:
        model_name_used = model_type
    else:
        model_name_used = args.model
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
        log(f"🎯 Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        log("🎯 Using CPU")
    
    # Load features with selection
    features_dict, unscaled_dict, pairs, labels, cluster_info, input_dim, feature_names = \
        load_features_with_selection(args.data_dir, args)
    
    # Create experiment name
    if args.exp_name:
        exp_name = args.exp_name
    else:
        feature_mode = "baseline" if args.baseline else "all"
        exp_name = f"{model_name_used}_{feature_mode}_h{args.hidden_dim}_l{args.layers}"
        if model_name_used in ['gat', 'transformer']:
            exp_name += f"_heads{args.heads}"
        if args.weighted_loss:
            exp_name += f"_{args.weight_strategy}"
    
    model_filename = f"{exp_name}.pt"
    
    log("\n" + "="*60)
    log(f"🔬 EXPERIMENT: {exp_name}")
    log("="*60)
    log(f"   Model: {model_name_used.upper()}")
    if len(feature_names) > 5:
        log(f"   Features: {input_dim} ({feature_names[:5]}...)")
    else:
        log(f"   Features: {input_dim} ({feature_names})")
    log(f"   Hidden dim: {args.hidden_dim}")
    log(f"   Layers: {args.layers}")
    if model_name_used in ['gat', 'transformer']:
        log(f"   Attention heads: {args.heads}")
    log(f"   Dropout: {args.dropout}")
    log(f"   Learning rate: {args.lr}")
    log(f"   Weight decay: {args.weight_decay}")
    log(f"   Train ratio: {args.train_ratio}")
    log("="*60)
    
    # Create loss function
    criterion = create_loss_function(args, labels, device)
    
    # Create model
    model = GraphFoundationModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=5,
        device=device,
        model_type=model_name_used,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        layer_weights=args.layer_weights,
        softmax_weights=args.softmax_weights,
        norm_type=args.norm,
        debug=args.debug
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"   Trainable parameters: {num_params:,}")
    
    # ============ INFERENCE-ONLY MODE ============
    if args.inference_only:
        log("\n" + "="*60)
        log("🔮 INFERENCE-ONLY MODE")
        log("="*60)
        
        # Find best model
        best_model_path = os.path.join(args.save_dir, f"best_{model_filename}")
        if not os.path.exists(best_model_path):
            log(f"❌ Best model not found: {best_model_path}")
            log("   Looking for any checkpoint...")
            chk = find_latest_checkpoint(args.save_dir, model_filename)
            if chk:
                _, best_model_path = chk
                log(f"   Using checkpoint: {best_model_path}")
            else:
                log("❌ No checkpoint found. Cannot run inference-only.")
                return None, None
        
        # Load best model
        log(f"📂 Loading model: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        epoch_loaded = checkpoint.get('epoch', 'unknown')
        log(f"   Loaded model from epoch {epoch_loaded}")
        
        # Create test generator
        gen_kwargs = {
            'features_dict': features_dict,
            'neighbor_pairs': pairs,
            'labels': labels,
            'unscaled_data_dict': unscaled_dict,
            'cluster_info_dict': cluster_info,
            'debug': args.debug,
            'is_bi_directional': True,
            'train_ratio': args.train_ratio,
        }
        test_generator = MultiClassBatchGenerator(mode='test', **gen_kwargs)
        
        # Create DataLoader for evaluation
        test_loader = DataLoader(
            test_generator,
            batch_size=args.batch_size,
            collate_fn=MultiClassBatchGenerator.collate_data,
            pin_memory=True,
            num_workers=0
        )
        
        # Create criterion (needed for loss in evaluate, but we only care about metrics)
        criterion = nn.CrossEntropyLoss()
        
        # ===== RUN EVALUATION WITH COMPREHENSIVE METRICS =====
        log("\n🔮 Running evaluation on test set...")
        t0 = time.perf_counter()
        test_res = evaluate(model, test_loader, criterion, device, use_amp=False)
        elapsed = time.perf_counter() - t0
        log(f"   ✅ Evaluation complete in {elapsed:.1f}s")
        
        # ===== ALSO RUN PER-EVENT INFERENCE FOR PARQUET =====
        log("\n💾 Running per-event inference for Parquet export...")
        final_results = run_inference(model, test_generator, device, args.debug, show_progress=True)
        
        # ===== SAVE PARQUET =====
        model_base = os.path.splitext(model_filename)[0]
        parquet_path = os.path.join(args.save_dir, f"results_{model_base}.parquet")
        save_results_to_parquet(final_results, parquet_path, model_base, cluster_info)
        
        # ===== CREATE COMPREHENSIVE METRICS (WITH ARGS!) =====
        metrics = {
            'best_epoch': epoch_loaded,
            'best_accuracy': test_res['accuracy'],
            'best_macro_recall': test_res['macro_recall'],
            'best_macro_precision': test_res['macro_precision'],
            'best_macro_f1': test_res['macro_f1'],
            'best_weighted_recall': test_res['weighted_recall'],
            'best_weighted_precision': test_res['weighted_precision'],
            'best_weighted_f1': test_res['weighted_f1'],
            'best_randomness_metric': test_res['randomness_metric'],
            'best_f1_sum_score': test_res['f1_sum_score'],
            'best_weighted_recall_score': test_res['weighted_recall_score'],
            'best_weighted_f1_score': test_res['weighted_f1_score'],
            'num_events_evaluated': len(final_results),
            'parquet_results_path': parquet_path,
            'model_path': best_model_path,
            # ===== CRITICAL: Save args so comparison script knows architecture & loss =====
            'args': vars(args),
        }
        
        # Add per-class metrics
        for c in range(5):
            metrics[f'test_recall_class_{c}'] = [test_res.get(f'recall_class_{c}', 0.0)]
            metrics[f'test_precision_class_{c}'] = [test_res.get(f'precision_class_{c}', 0.0)]
            metrics[f'test_f1_class_{c}'] = [test_res.get(f'f1_class_{c}', 0.0)]
        
        # Save PKL
        metrics_path = os.path.join(args.save_dir, f"{model_base}_metrics.pkl")
        
        # Merge with existing metrics if available
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'rb') as f:
                    existing_metrics = pickle.load(f)
                for key in metrics:
                    existing_metrics[key] = metrics[key]
                metrics = existing_metrics
            except:
                pass
        
        save_pickle(metrics, metrics_path)
        
        # ===== SUMMARY =====
        log("\n" + "="*60)
        log("✅ INFERENCE COMPLETE!")
        log("="*60)
        log(f"   Accuracy:           {test_res['accuracy']:.4f}")
        log(f"   Macro F1:           {test_res['macro_f1']:.4f}")
        log(f"   Macro Recall:       {test_res['macro_recall']:.4f}")
        log(f"   Macro Precision:    {test_res['macro_precision']:.4f}")
        log(f"   F1 Sum Score (FSS): {test_res['f1_sum_score']:.2f}  (1.0=random, 5.0=perfect)")
        log(f"   Recall Sum (RSS):   {test_res['randomness_metric']:.2f}  (1.0=random, 5.0=perfect)")
        log(f"   Weighted F1 (WFSS): {test_res['weighted_f1_score']:.2f}")
        log(f"")
        log(f"   Per-Class Breakdown:")
        for c in range(5):
            rec = test_res.get(f'recall_class_{c}', 0.0)
            prec = test_res.get(f'precision_class_{c}', 0.0)
            f1c = test_res.get(f'f1_class_{c}', 0.0)
            log(f"     Class {c}: Recall={rec:.4f}, Precision={prec:.4f}, F1={f1c:.4f}")
        log(f"\n   Parquet saved to: {parquet_path}")
        log(f"   Metrics saved to: {metrics_path}")
        log("="*60)
        
        return metrics, best_model_path

    # Normal training mode
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scaler for mixed precision
    if args.mixed_precision and args.gpu >= 0:
        scaler = torch.amp.GradScaler('cuda')
        log("✅ Mixed precision enabled - using FP16 with GradScaler")
    else:
        scaler = None
        log("✅ Mixed precision disabled - using FP32")   
   
    # Generator kwargs
    gen_kwargs = {
        'features_dict': features_dict,
        'neighbor_pairs': pairs,
        'labels': labels,
        'unscaled_data_dict': unscaled_dict,
        'cluster_info_dict': cluster_info,
        'debug': args.debug,
        'is_bi_directional': True,
        'train_ratio': args.train_ratio,
    }
    
    # Create generators and loaders
    train_generator = MultiClassBatchGenerator(mode='train', **gen_kwargs)
    test_generator = MultiClassBatchGenerator(mode='test', **gen_kwargs)
    
    train_loader = DataLoader(
        train_generator,
        batch_size=args.batch_size,
        collate_fn=MultiClassBatchGenerator.collate_data,
        pin_memory=True,
        num_workers=0  # Keep at 0 due to CUDA multiprocessing issues
    )
    
    test_loader = DataLoader(
        test_generator,
        batch_size=args.batch_size,
        collate_fn=MultiClassBatchGenerator.collate_data,
        pin_memory=True,
        num_workers=0
    )
    
    # Train
    metrics, model, model_path = train_model_full(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        test_generator=test_generator,
        optimizer=optimizer,
        criterion=criterion,
        scaler=scaler,
        device=device,
        args=args,
        model_name=model_filename,
        cluster_info=cluster_info
    )
    
    # Summary
    log("\n" + "="*60)
    log("🏁 TRAINING SUMMARY")
    log("="*60)
    log(f"   Best epoch: {metrics['best_epoch']}")
    log(f"   Best ACCURACY: {metrics['best_accuracy']:.4f}")
    log(f"   Best MACRO F1: {metrics['best_macro_f1']:.4f}")
    log(f"   Best MACRO RECALL: {metrics['best_macro_recall']:.4f}")
    log(f"   Best MACRO PRECISION: {metrics['best_macro_precision']:.4f}")
    log(f"   Best WEIGHTED F1: {metrics['best_weighted_f1']:.4f}")
    log(f"")
    log(f"   Best F1 SUM SCORE (FSS): {metrics['best_f1_sum_score']:.2f}  (1.0=random, 5.0=perfect)")
    log(f"   Best RECALL SUM SCORE (RSS): {metrics['best_randomness_metric']:.2f}  (1.0=random, 5.0=perfect)")
    log(f"   Best WEIGHTED F1 SCORE (WFSS): {metrics['best_weighted_f1_score']:.2f}")
    log(f"")
    log(f"   Best per-class metrics:")
    for c in range(5):
        recall_c = max(metrics.get(f'test_recall_class_{c}', [0]))
        prec_c = max(metrics.get(f'test_precision_class_{c}', [0]))
        f1_c = max(metrics.get(f'test_f1_class_{c}', [0]))
        log(f"     Class {c}: Recall={recall_c:.4f}, Precision={prec_c:.4f}, F1={f1_c:.4f}")
    log(f"\n   Best test loss: {metrics['best_test_loss']:.6f}")
    log(f"   Total time: {metrics['total_time']/60:.1f} min")
    log(f"   Model saved to: {model_path}")
    if 'parquet_results_path' in metrics:
        log(f"   Results saved to: {metrics['parquet_results_path']}")
    log("="*60)

    return metrics, model_path

# ============================================================================
# CONVERSION FUNCTION
# ============================================================================

def convert_results(pickle_path: str) -> None:
    """Convert old pickle results to new Parquet format."""
    log(f"Converting {pickle_path} to Parquet format...")
    
    # Load pickle
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract results
    if 'final_results_per_event' in data:
        results = data['final_results_per_event']
    else:
        results = data
    
    # Convert to DataFrame
    all_dfs = []
    for result in results:
        num_edges = len(result.get('preds', []))
        if num_edges == 0:
            continue
            
        df = pd.DataFrame({
            'event_id': result.get('event_id', 0),
            'pred_label': result.get('preds', []),
            'true_label': result.get('labels', []) if result.get('labels') is not None else -1,
        })
        all_dfs.append(df)
    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        output_path = pickle_path.replace('.pkl', '.parquet').replace('.pickle', '.parquet')
        final_df.to_parquet(output_path, compression='snappy')
        log(f"✅ Converted to: {output_path}")
        log(f"   Total edges: {len(final_df):,}")
    else:
        log("❌ No results found to convert")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    log("="*70)
    log("🚀 GRAPH FOUNDATION MODEL FOR PARTICLE PHYSICS")
    log("="*70)
    log(f"Model: {args.model.upper()}")
    log(f"Features: {'BASELINE (3)' if args.baseline else 'ALL (42+)'}")
    log(f"Hidden dim: {args.hidden_dim}, Layers: {args.layers}")
    log(f"GPU: {args.gpu if args.gpu >= 0 else 'CPU'}")
    log("="*70)
    
    if args.model == 'all':
        # Train all architectures sequentially
        architectures = ['gcn', 'gat', 'transformer', 'sage']
        log(f"\n🔄 Training all architectures: {architectures}")
        
        results = {}
        for arch in architectures:
            log(f"\n{'='*60}")
            log(f"Training {arch.upper()}...")
            log('='*60)
            
            try:
                metrics, model_path = train_single_model(args, model_type=arch)
                results[arch] = {
                    'best_acc': metrics['best_test_acc'],
                    'model_path': model_path
                }
                log(f"✅ {arch.upper()} completed! Best acc: {metrics['best_test_acc']:.4f}")
            except Exception as e:
                log(f"❌ {arch.upper()} failed: {e}")
                traceback.print_exc()
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            time.sleep(5)
        
        # Summary
        log("\n" + "="*60)
        log("📊 ALL MODELS SUMMARY")
        log("="*60)
        for arch, res in results.items():
            log(f"   {arch.upper()}: Best acc = {res['best_acc']:.4f}")
        log("="*60)
        
    else:
        # Train single architecture
        try:
            metrics, model_path = train_single_model(args)
            log(f"\n✅ Training completed!")
        except Exception as e:
            log(f"❌ Training failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    log("\n" + "="*70)
    log("🎉 PIPELINE COMPLETE!")
    log("="*70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check for convert mode
    if len(sys.argv) > 1 and sys.argv[1] == 'convert':
        if len(sys.argv) < 3:
            print("Usage: python3 new_RunningMultipleModels.py convert path/to/results.pkl")
            sys.exit(1)
        convert_results(sys.argv[2])
    else:
        main()
