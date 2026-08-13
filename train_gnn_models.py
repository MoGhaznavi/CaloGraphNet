#!/usr/bin/env python3
"""
Graph Foundation Model for Particle Physics Edge Classification

CUDA/LINUX VERSION - FULLY INTEGRATED & OPTIMIZED
"""

# ============================================================================
# IMPORTS
# ============================================================================

import argparse
import datetime
import gc
import glob
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
import pickle
import psutil
import re
import signal
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.nn import BatchNorm, LayerNorm
from torch.nn import BatchNorm1d, LayerNorm as LayerNorm1d
from torch_geometric.utils import to_undirected
import pyarrow as pa
import pyarrow.parquet as pq
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
# UTILITY FUNCTIONS
# ============================================================================

def log(msg: str) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


# ============================================================================
# GLOBAL SETUP
# ============================================================================

if mp.get_start_method(allow_none=True) != 'spawn':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)


# ============================================================================
# RESOURCE TRACKER CLASS
# ============================================================================

class ResourceTracker:
    """Tracks runtime, memory, CPU, and optional GPU usage during execution."""

    def __init__(self, log_dir: str = None, enabled: bool = True):
        # Enable/disable resource tracking entirely.
        self.enabled = enabled

        # Directory where resource reports will be written.
        self.log_dir = log_dir or './resource_logs'

        # Create the logging directory if tracking is enabled.
        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)

        # Store all measurements collected during execution.
        self.measurements = []

        # psutil process object for querying resource usage.
        self.process = psutil.Process() if self.enabled else None

        # Baseline measurements recorded when start() is called.
        self.start_time = None
        self.start_memory = None
        self.start_gpu_memory = None

    def start(self):
        """Record baseline resource usage before execution begins."""
        if not self.enabled:
            return

        # Start timing.
        self.start_time = time.perf_counter()

        # Record initial RAM usage (GB).
        self.start_memory = self.process.memory_info().rss / 1024**3

        # Record initial GPU memory usage if CUDA is available.
        if torch.cuda.is_available():
            self.start_gpu_memory = torch.cuda.memory_allocated() / 1024**3
            # ---- FIX: clear any peak-memory history from before start()
            # was called, so the first measure() call's peak reflects only
            # what happens after this point.
            torch.cuda.reset_peak_memory_stats()

    def measure(self, stage: str) -> Optional[Dict]:
        """
        Record the current resource usage.

        Parameters
        ----------
        stage : str
            Descriptive name of the current execution stage.
        """
        if not self.enabled:
            return None

        # Current RAM usage (GB).
        mem = self.process.memory_info().rss / 1024**3

        # Elapsed wall-clock time since start().
        elapsed = time.perf_counter() - self.start_time

        # Store the primary measurements.
        measurement = {
            'stage': stage,
            'timestamp': datetime.datetime.now().isoformat(),
            'time_elapsed': elapsed,
            'memory_gb': mem,
            'memory_delta_gb': mem - self.start_memory,
        }

        # ---- FIX: report PEAK GPU usage since the last reset, not the
        # instantaneous live-tensor count at this exact snapshot moment.
        # memory_allocated() only counts tensors PyTorch's allocator
        # currently considers live — between epochs, after batch tensors
        # have gone out of scope, that number collapses toward zero even
        # though the epoch just did heavy GPU work throughout. The peak
        # counters (max_memory_allocated / max_memory_reserved) track the
        # high-water mark since the last reset_peak_memory_stats() call,
        # which is what you actually want to see here.
        if torch.cuda.is_available():
            measurement['gpu_memory_current_gb'] = torch.cuda.memory_allocated() / 1024**3
            measurement['gpu_memory_peak_gb'] = torch.cuda.max_memory_allocated() / 1024**3
            measurement['gpu_memory_reserved_peak_gb'] = torch.cuda.max_memory_reserved() / 1024**3
            measurement['gpu_memory_delta_gb'] = (
                measurement['gpu_memory_current_gb'] - self.start_gpu_memory
            )
            measurement['gpu_memory_type'] = 'cuda'

            # Reset the peak counters so the NEXT measurement's peak
            # reflects only the interval between this call and the next,
            # rather than accumulating across the whole run.
            torch.cuda.reset_peak_memory_stats()

        # Collect additional process statistics when available.
        try:
            measurement['cpu_percent'] = self.process.cpu_percent()

            io_counters = self.process.io_counters()
            measurement['disk_read_gb'] = io_counters.read_bytes / 1024**3
            measurement['disk_write_gb'] = io_counters.write_bytes / 1024**3

        # Some platforms do not expose all process statistics.
        except:
            pass

        # Save the measurement for later reporting.
        self.measurements.append(measurement)

        return measurement

    def log_measurement(self, stage: str, extra_info: str = ""):
        """
        Measure the current resource usage and print a concise summary.
        """
        m = self.measure(stage)

        if m:
            gpu_str = (
                f" | GPU peak: {m['gpu_memory_peak_gb']:.2f}GB "
                f"(reserved peak: {m['gpu_memory_reserved_peak_gb']:.2f}GB)"
                if 'gpu_memory_peak_gb' in m else ""
            )

            log(
                f"  📊 [{stage}] "
                f"Time: {m['time_elapsed']:.1f}s | "
                f"RAM: {m['memory_gb']:.2f}GB "
                f"(Δ{m['memory_delta_gb']:+.2f})"
                f"{gpu_str}"
                + (f" | {extra_info}" if extra_info else "")
            )

        return m

    def save_report(self, filename: str = "resource_report.json"):
        """Save all recorded measurements to a JSON report."""
        if not self.enabled or not self.measurements:
            return

        filepath = os.path.join(self.log_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.measurements, f, indent=2, default=str)

        log(f"📊 Resource report saved to: {filepath}")

    def print_summary(self):
        """Print a summary of the overall resource usage."""
        if not self.enabled or not self.measurements:
            return

        # Total runtime corresponds to the last recorded measurement.
        total_time = self.measurements[-1]['time_elapsed']

        # Peak RAM usage across all measurements.
        peak_memory = max(m['memory_gb'] for m in self.measurements)

        log(f"\n📊 RESOURCE USAGE SUMMARY:")
        log(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        log(f"   Peak RAM: {peak_memory:.2f} GB")

# ============================================================================
# ARGPARSE CONFIGURATION
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Graph Foundation Model for Particle Physics Edge Classification')
    parser.add_argument('--model', '-m', type=str, default='gcn', choices=['gcn', 'gat', 'transformer', 'sage', 'all'])
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--layers', '-l', type=int, default=6)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--layer-weights', action='store_true')
    parser.add_argument('--softmax-weights', action='store_true')
    parser.add_argument('--norm', type=str, default='batch', choices=['batch', 'layer', 'none'])
    parser.add_argument('--baseline', action='store_true', default=True)
    parser.add_argument('--all-features', action='store_true')
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--weighted-loss', action='store_true')
    parser.add_argument('--weight-strategy', type=str, default='inverse', choices=['inverse', 'focal', 'logarithmic', 'manual'])
    parser.add_argument('--focal-alpha', type=float, default=0.25)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--mixed-precision', action='store_true', default=True)
    parser.add_argument('--no-mixed-precision', action='store_false', dest='mixed_precision')
    parser.add_argument('--save-dir', type=str, default='./experiments')
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--inference-only', action='store_true')
    parser.add_argument('--analyze-scalability', action='store_true')
    parser.add_argument('--track-resources', action='store_true', default=True)
    parser.add_argument('--no-track-resources', action='store_false', dest='track_resources')
    parser.add_argument('--pretrain', action='store_true', help='Enable masked pretraining mode')
    parser.add_argument('--mask-type', type=str, default='random', choices=['random', 'feature', 'geometry', 'cluster'], help='Type of masking strategy')
    parser.add_argument('--mask-ratio', type=float, default=0.15, help='Fraction of cells/features to mask')
    parser.add_argument('--mask-features', type=str, nargs='+', default=None, help='Specific features to mask (for feature masking)')
    parser.add_argument('--geometry-radius', type=int, default=2, help='Radius for geometry masking')
    parser.add_argument('--pretrain-epochs', type=int, default=100, help='Number of pretraining epochs')
    parser.add_argument('--finetune-epochs', type=int, default=30, help='Number of finetuning epochs (after pretraining)')
    parser.add_argument('--continuous-loss', type=str, default='mse', choices=['mse', 'l1'], help='Loss function for continuous features')
    return parser.parse_args()


# ============================================================================
# UTILITY FUNCTIONS (continued)
# ============================================================================

def save_pickle(data: Any, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filepath: str) -> Any:
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def find_latest_checkpoint(save_dir: str, base_name: str) -> Optional[Tuple[int, str]]:
    pattern = re.compile(f"{re.escape(os.path.splitext(base_name)[0])}_epoch(\\d+).pt")
    checkpoints = []
    for fname in os.listdir(save_dir):
        match = pattern.match(fname)
        if match:
            checkpoints.append((int(match.group(1)), os.path.join(save_dir, fname)))
    return max(checkpoints, key=lambda x: x[0]) if checkpoints else None

def analyze_dataset_scalability(data_dir: str) -> Dict:
    log(f"\n📊 ANALYZING DATASET SCALABILITY: {data_dir}")
    log("="*70)
    results = {'directory': data_dir, 'files': {}, 'scalability': {}}
    all_files = glob.glob(os.path.join(data_dir, "*"))
    total_size_gb = 0
    for filepath in all_files:
        if os.path.isfile(filepath):
            filename = os.path.basename(filepath)
            size_gb = os.path.getsize(filepath) / 1024**3
            total_size_gb += size_gb
            if 'events_' in filename and filename.endswith('.h5'):
                file_type = 'event_data'
            elif 'labels_' in filename and filename.endswith('.npy'):
                file_type = 'labels'
            elif 'pairs_' in filename and filename.endswith('.npy'):
                file_type = 'pairs'
            elif 'cells_' in filename and filename.endswith('.npy'):
                file_type = 'cells'
            elif filename.endswith('.json'):
                file_type = 'metadata'
            elif filename.endswith('.pkl'):
                file_type = 'scaler'
            else:
                file_type = 'other'
            if file_type not in results['files']:
                results['files'][file_type] = {'count': 0, 'total_size_gb': 0, 'files': []}
            results['files'][file_type]['count'] += 1
            results['files'][file_type]['total_size_gb'] += size_gb
            results['files'][file_type]['files'].append({'name': filename, 'size_gb': size_gb})
    
    total_events = None
    metadata_files = glob.glob(os.path.join(data_dir, "metadata_*.json"))
    if metadata_files:
        with open(metadata_files[0], 'r') as f:
            total_events = json.load(f).get('total_events')
    if not total_events:
        label_files = sorted(glob.glob(os.path.join(data_dir, "labels_*.npy")))
        if label_files:
            total_events = sum(np.lib.format.read_array_header_1_0(np.lib.format.read_magic(open(lf,'rb')))[0][0] for lf in label_files)
    
    if total_events:
        results['scalability']['total_events'] = total_events
        for scale in [10_000, 50_000, 100_000, 500_000, 1_000_000]:
            total_gb_scaled = sum(info['total_size_gb'] / total_events * scale for info in results['files'].values() if info['count'] > 0)
            results['scalability'][f'{scale}_events_gb'] = total_gb_scaled
    
    log(f"\n📁 FILE BREAKDOWN: Total {total_size_gb:.2f} GB")
    for file_type, info in sorted(results['files'].items()):
        log(f"   {file_type:<15s}: {info['count']:3d} files, {info['total_size_gb']:8.2f} GB")
    if 'scalability' in results:
        log(f"\n📈 SCALABILITY:")
        for scale in [10_000, 50_000, 100_000, 500_000, 1_000_000]:
            if f'{scale}_events_gb' in results['scalability']:
                log(f"   {scale:,} events: {results['scalability'][f'{scale}_events_gb']:.1f} GB")
    return results

# ============================================================================
# MASKING FUNCTIONS FOR PRETRAINING
# ============================================================================

class CalorimeterMasking:
    """
    Implements various masking strategies for calorimeter cell graphs.
    
    Supports:
    - Random cell masking (like BERT)
    - Feature masking (mask specific features)
    - Geometry masking (mask contiguous regions)
    - Cluster masking (mask entire topo-clusters)
    """
    
    def __init__(self, mask_ratio=0.15, mask_type='random', 
                 mask_features=None, cells_array=None, 
                 cluster_info_dict=None, geometry_radius=2):
        """
        Args:
            mask_ratio: Fraction of cells/features to mask (0.0 to 1.0)
            mask_type: 'random', 'feature', 'geometry', 'cluster'
            mask_features: List of feature names to mask (for feature masking)
            cells_array: numpy structured array with cell positions (eta, phi)
            cluster_info_dict: Dict mapping event_id -> cluster_index array
            geometry_radius: Number of hops for geometry masking
        """
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.mask_features = mask_features or []
        self.cells_array = cells_array
        self.cluster_info_dict = cluster_info_dict or {}
        self.geometry_radius = geometry_radius
        
        # Learnable mask token (initialized as zeros, can be made learnable)
        self.mask_token_value = 0.0
        
    def _get_random_mask(self, num_cells: int, rng: np.random.RandomState) -> np.ndarray:
        """Random cell masking - mask entire cells."""
        mask = rng.random(num_cells) < self.mask_ratio
        return mask
    
    def _get_feature_mask(self, num_cells: int, num_features: int, 
                          feature_names: List[str], 
                          rng: np.random.RandomState) -> np.ndarray:
        """Feature masking - mask specific features across cells."""
        # Create a boolean mask of shape (num_cells, num_features)
        mask = np.zeros((num_cells, num_features), dtype=bool)
        
        if not self.mask_features:
            # If no features specified, randomly choose features
            n_features_to_mask = max(1, int(num_features * self.mask_ratio))
            features_to_mask = rng.choice(num_features, n_features_to_mask, replace=False)
        else:
            # Map feature names to indices
            features_to_mask = []
            for fname in self.mask_features:
                if fname in feature_names:
                    features_to_mask.append(feature_names.index(fname))
            if not features_to_mask:
                # Fallback: mask a random feature
                features_to_mask = [rng.randint(0, num_features)]
        
        # Mask the selected features for a subset of cells
        cells_to_mask = rng.random(num_cells) < self.mask_ratio
        for feat_idx in features_to_mask:
            mask[cells_to_mask, feat_idx] = True
            
        return mask
    
    def _get_geometry_mask(self, num_cells: int, cell_positions: np.ndarray,
                          rng: np.random.RandomState) -> np.ndarray:
        """Geometry masking - mask cells in a contiguous spatial region."""
        mask = np.zeros(num_cells, dtype=bool)
        
        # Choose a random seed cell
        seed_cell = rng.randint(0, num_cells)
        seed_eta = cell_positions[seed_cell, 0]
        seed_phi = cell_positions[seed_cell, 1]
        
        # Mask cells within a certain eta-phi window
        # Note: phi wrapping not handled here for simplicity
        # In practice, you'd want proper angular distance
        eta_window = 0.3 * self.geometry_radius  # Adjust window size
        phi_window = 0.3 * self.geometry_radius
        
        for i in range(num_cells):
            deta = abs(cell_positions[i, 0] - seed_eta)
            dphi = abs(cell_positions[i, 1] - seed_phi)
            # Handle phi wrapping
            dphi = min(dphi, 2*np.pi - dphi)
            
            if deta < eta_window and dphi < phi_window:
                # Only mask a fraction to avoid too easy reconstruction
                if rng.random() < self.mask_ratio * 2:  # Higher probability in window
                    mask[i] = True
                    
        return mask
    
    def _get_cluster_mask(self, num_cells: int, event_id: int,
                         rng: np.random.RandomState) -> np.ndarray:
        """Cluster masking - mask all cells belonging to selected clusters."""
        mask = np.zeros(num_cells, dtype=bool)
        
        if event_id not in self.cluster_info_dict:
            # Fallback to random masking
            return self._get_random_mask(num_cells, rng)
        
        cluster_info = self.cluster_info_dict[event_id]
        cluster_ids = cluster_info.get('cell_cluster_index', None)
        
        if cluster_ids is None:
            return self._get_random_mask(num_cells, rng)
        
        # Get unique cluster IDs (excluding -1 for unclustered)
        unique_clusters = np.unique(cluster_ids[cluster_ids >= 0])
        
        if len(unique_clusters) == 0:
            return self._get_random_mask(num_cells, rng)
        
        # Select clusters to mask
        n_clusters_to_mask = max(1, int(len(unique_clusters) * self.mask_ratio))
        clusters_to_mask = rng.choice(unique_clusters, n_clusters_to_mask, replace=False)
        
        # Mask all cells in selected clusters
        for cluster_id in clusters_to_mask:
            mask[cluster_ids == cluster_id] = True
            
        return mask
    
    def apply_mask(self, features: torch.Tensor, event_id: int = None,
                  feature_names: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply masking to node features.
        
        Args:
            features: Node features tensor of shape (num_cells, num_features)
            event_id: Event identifier (needed for cluster masking)
            feature_names: List of feature names (needed for feature masking)
            
        Returns:
            masked_features: Features with masked values replaced
            mask: Boolean mask indicating which values were masked
            targets: Original values for masked positions (for reconstruction loss)
        """
        rng = np.random.RandomState()
        num_cells, num_features = features.shape
        
        if self.mask_type == 'random':
            # Cell-level mask
            cell_mask = self._get_random_mask(num_cells, rng)
            # Expand to feature dimension
            mask = np.tile(cell_mask[:, np.newaxis], (1, num_features))
            
        elif self.mask_type == 'feature':
            # Feature-level mask
            mask = self._get_feature_mask(num_cells, num_features, feature_names or [], rng)
            
        elif self.mask_type == 'geometry':
            # Geometry-based mask
            if self.cells_array is not None:
                # Extract eta, phi positions
                cell_positions = np.column_stack([
                    self.cells_array['eta_event0'] if 'eta_event0' in self.cells_array.dtype.names 
                    else self.cells_array['eta'],
                    self.cells_array['phi_event0'] if 'phi_event0' in self.cells_array.dtype.names
                    else self.cells_array['phi']
                ])
                cell_mask = self._get_geometry_mask(num_cells, cell_positions, rng)
            else:
                cell_mask = self._get_random_mask(num_cells, rng)
            mask = np.tile(cell_mask[:, np.newaxis], (1, num_features))
            
        elif self.mask_type == 'cluster':
            # Cluster-based mask
            cell_mask = self._get_cluster_mask(num_cells, event_id or 0, rng)
            mask = np.tile(cell_mask[:, np.newaxis], (1, num_features))
            
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")
        
        # Convert to tensors
        mask = torch.from_numpy(mask).bool()
        
        # Store original values as targets (only for masked positions)
        targets = features.clone()
        
        # Apply masking (replace with mask token value)
        masked_features = features.clone()
        masked_features[mask] = self.mask_token_value
        
        return masked_features, mask, targets
    
    def get_reconstruction_mask(self, mask: torch.Tensor, 
                                targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get only the masked positions for loss computation.
        
        Returns:
            masked_targets: Target values for masked positions
            valid_mask: Boolean mask for positions to include in loss
        """
        # Only compute loss on masked positions
        return targets[mask], mask

# ============================================================================
# FEATURE LOADING
# ============================================================================

def load_features_lazy(data_dir: str, args, tracker: Optional[ResourceTracker] = None) -> Tuple:
    """
    Lazy feature loading: returns HDF5 references instead of pre-loading all events.
    Labels are memory-mapped for zero-copy access.
    """
    log(f"📊 Lazy feature loading: {'ALL FEATURES' if args.all_features else 'BASELINE (7 features)'}")
    log(f"📁 Data directory: {data_dir}")
    if tracker:
        tracker.log_measurement("start_loading")
    
    cells_files = sorted(glob.glob(os.path.join(data_dir, "cells_*.npy")))
    pairs_files = sorted(glob.glob(os.path.join(data_dir, "pairs_*.npy")))
    event_files = sorted(glob.glob(os.path.join(data_dir, "events_*.h5")))
    label_files = sorted(glob.glob(os.path.join(data_dir, "labels_*.npy")))
    metadata_files = glob.glob(os.path.join(data_dir, "metadata_*.json"))
    
    if not event_files:
        old = os.path.join(data_dir, "events.h5")
        if os.path.exists(old): event_files = [old]
    if not label_files:
        old = os.path.join(data_dir, "labels.npy")
        if os.path.exists(old): label_files = [old]
    if not cells_files: cells_files = [os.path.join(data_dir, "cells.npy")]
    if not pairs_files: pairs_files = [os.path.join(data_dir, "pairs.npy")]
    
    log(f"\n📂 FILES: {len(cells_files)} cells, {len(pairs_files)} pairs, {len(event_files)} events, {len(label_files)} labels")
    total_size_gb = sum(os.path.getsize(f)/1024**3 for f in cells_files+pairs_files+event_files+label_files if os.path.exists(f))
    log(f"   Total: {total_size_gb:.2f} GB")
    
    # Load static data (small, fits in RAM)
    cells = np.load(cells_files[0])
    num_cells = cells.shape[0]
    log(f"  ✓ Cells: {num_cells} ({cells.nbytes/1024**2:.1f} MB)")
    
    pairs = np.load(pairs_files[0]).astype(np.int32)
    num_edges = pairs.shape[0]
    log(f"  ✓ Pairs: {num_edges} ({pairs.nbytes/1024**2:.1f} MB)")
    if tracker: tracker.log_measurement("static_files_loaded")
        
    # Load labels into RAM (12 GB, manageable)
    log(f"\n📊 Loading labels...")
    label_chunks = [np.load(lf).astype(np.int8) for lf in label_files]
    labels = np.concatenate(label_chunks) if len(label_chunks) > 1 else label_chunks[0]
    log(f"  ✓ Labels: {labels.shape} ({labels.nbytes/1024**2:.0f} MB)")
    del label_chunks; gc.collect()
    if tracker: tracker.log_measurement("labels_mapped")
    
    if args.baseline and not args.all_features:
        feature_names = ['snr_scaled', 'snr_gt4', 'snr_gt2', 'snr_gt0',
                         'eta', 'sin_phi', 'cos_phi']
        input_dim = 7
    else:
        feature_names = []
        input_dim = 0
    
    # Build feature references instead of loading all data
    feature_refs = []  # List of (hdf5_path, local_idx, has_eta, has_snr)
    
    for file_idx, event_file in enumerate(event_files):
        with h5py.File(event_file, 'r') as h5f:
            if 'cell/snr_computed' in h5f:
                file_num_events = h5f['cell/snr_computed'].shape[0]
            elif 'cell/energy_raw' in h5f:
                file_num_events = h5f['cell/energy_raw'].shape[0]
            else:
                total_label_events = sum(arr.shape[0] for arr in labels)
                file_num_events = total_label_events // len(event_files) if len(event_files) > 1 else total_label_events
            
            has_eta = 'cell/cell_eta' in h5f
            has_snr = 'cell/snr_computed' in h5f or 'cell/snr_raw' in h5f or 'cell/energy_raw' in h5f
            
            for local_idx in range(file_num_events):
                feature_refs.append({
                    'hdf5_path': event_file,
                    'local_idx': local_idx,
                    'has_eta': has_eta,
                    'has_snr': has_snr,
                })
    
    if tracker:
        tracker.log_measurement("feature_refs_built", f"{len(feature_refs)} events, {input_dim} features")
    log(f"\n✅ Lazy references built: {len(feature_refs)} events, {num_cells} cells, {num_edges} edges, {input_dim} features")
    log(f"   RAM usage: ~{cells.nbytes/1024**2:.0f} MB (cells) + ~{pairs.nbytes/1024**2:.0f} MB (pairs) + ~{labels.nbytes/1024**2:.0f} MB (labels)")
    return feature_refs, pairs, labels, None, input_dim, feature_names, cells


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean', chunk_size=100000):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.reduction = reduction; self.chunk_size = chunk_size
        self.alpha = float(alpha) if isinstance(alpha, (float,int)) else None
        self.per_class_alpha = torch.tensor(alpha, dtype=torch.float32) if isinstance(alpha, list) else None
            
    def forward(self, inputs, targets):
        total_loss = 0.0
        for start in range(0, inputs.size(0), self.chunk_size):
            end = min(start+self.chunk_size, inputs.size(0))
            ce = F.cross_entropy(inputs[start:end], targets[start:end], weight=self.weight, reduction='none')
            pt = torch.exp(-ce)
            alpha_t = self.per_class_alpha.to(inputs.device)[targets[start:end]] if self.per_class_alpha is not None else self.alpha
            total_loss += (alpha_t * (1-pt)**self.gamma * ce).sum()
        return total_loss/inputs.size(0) if self.reduction=='mean' else total_loss

def compute_class_weights(labels, num_classes, strategy='inverse', device=None, **kwargs):
    if device and device.type=='cuda':
        counts = torch.bincount(torch.as_tensor(labels, device=device).flatten(), minlength=num_classes).float()
    else:
        counts = torch.tensor(np.bincount(labels.flatten(), minlength=num_classes), dtype=torch.float32)
    log(f"Class counts: {dict(zip(range(num_classes), counts.int().tolist()))}")
    
    if strategy=='focal':
        total = len(labels.flatten())
        weights = kwargs.get('alpha',0.25) * ((total-counts)/(counts+1e-5))**kwargs.get('gamma',2.0)
    elif strategy=='logarithmic': weights = 1.0/torch.log1p(counts+1e-5)
    elif strategy=='manual': weights = torch.tensor([0.1,10.0,8.0,8.0,15.0], device=counts.device)[:num_classes]
    else: weights = 1.0/(counts+1e-5)
    
    weights = weights/weights.sum()*num_classes
    log(f"Computed weights ({strategy}): {weights.tolist()}")
    return weights.to(device) if device else weights

def create_loss_function(args, labels, device):
    if args.weighted_loss:
        class_weights = compute_class_weights(labels, 5, strategy=args.weight_strategy, device=device, alpha=args.focal_alpha, gamma=args.focal_gamma)
        if args.weight_strategy=='focal':
            log(f"✅ Using Focal Loss")
            return FocalLoss(alpha=[0.10,0.60,0.70,0.70,1.00], gamma=args.focal_gamma)
        log(f"✅ Using Weighted CrossEntropyLoss")
        return nn.CrossEntropyLoss(weight=class_weights)
    log("✅ Using standard CrossEntropyLoss")
    return nn.CrossEntropyLoss()


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class GraphFoundationModel(nn.Module):
    """
    Generic graph neural network for edge classification and masked pretraining.
    
    Supports:
    - Multiple message-passing backbones (GCN, GAT, GraphSAGE, TransformerConv)
    - Optional learnable layer weighting
    - Masked pretraining with reconstruction head
    - Downstream edge classification
    """

    def __init__(self, input_dim, hidden_dim, output_dim, device,
                 model_type='gcn', num_layers=6, num_heads=2,
                 dropout=0.0, layer_weights=False,
                 softmax_weights=False, norm_type='batch', debug=False,
                 pretraining=False, feature_names=None):
        super().__init__()

        # Store model configuration
        self.device = device
        self.model_type = model_type
        self.num_layers = num_layers
        self.layer_weights_enabled = layer_weights
        self.softmax = softmax_weights
        self.pretraining = pretraining
        self.feature_names = feature_names or []

        # Project input node features into the hidden embedding space
        self.node_embedding = nn.Linear(input_dim, hidden_dim)

        # Construct the requested graph convolution backbone
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if model_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif model_type == 'gat':
                self.convs.append(
                    GATConv(hidden_dim, hidden_dim // num_heads,
                           heads=num_heads, dropout=dropout,
                           edge_dim=5)
                )
            elif model_type == 'transformer':
                self.convs.append(
                    TransformerConv(hidden_dim, hidden_dim // num_heads,
                                   heads=num_heads, dropout=dropout,
                                    edge_dim=5)
                )
            elif model_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        # Create normalization layers
        if norm_type == 'batch':
            self.bns = nn.ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        elif norm_type == 'layer':
            self.bns = nn.ModuleList([LayerNorm1d(hidden_dim) for _ in range(num_layers)])
        else:
            self.bns = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

        # Task-specific heads
        if pretraining:
            # Reconstruction head for masked pretraining
            self.reconstruction_head = FeatureReconstructionHead(
                hidden_dim, input_dim,
                feature_types=self._infer_feature_types()
            )
            self.fc = None  # No classification head during pretraining
        else:
            # Edge classification head for downstream tasks
            self.fc = nn.Linear(2 * hidden_dim, output_dim)
            self.reconstruction_head = None

        # Optional learnable weighting of each message-passing layer
        if layer_weights:
            self.layer_weights = nn.Parameter(torch.ones(num_layers))

        log(
            f"📐 Initialized {model_type.upper()} model: "
            f"input={input_dim}, hidden={hidden_dim}, layers={num_layers}"
            f"{' [PRETRAINING]' if pretraining else ' [FINETUNING]'}"
        )

    def _infer_feature_types(self) -> List[str]:
        """Infer feature types from feature names."""
        categorical_features = {'subcalo', 'sampling', 'noise_category'}
        feature_types = []
        for fname in self.feature_names:
            if fname in categorical_features or fname.startswith('subcalo_') or fname.startswith('sampling_'):
                feature_types.append('categorical')
            else:
                feature_types.append('continuous')
        return feature_types if feature_types else ['continuous']

    def encode(self, x_list, edge_index_list, edge_attr_list=None):
        """
        Encode node features into embeddings — batched across graphs.
    
        Assumes every graph in the list shares the same node count and
        edge topology (true here: same detector geometry every event),
        so graphs are concatenated along the node dimension and edge
        indices are offset accordingly, giving one large block-diagonal
        graph the GPU can process in a single set of kernel calls.
        """
        weights = (
            torch.softmax(self.layer_weights, dim=0)
            if self.softmax else self.layer_weights
        ) if self.layer_weights_enabled else None
    
        num_graphs = len(x_list)
        num_nodes = x_list[0].shape[0]
    
        # Concatenate all node features into one big batch.
        x_batch = torch.cat(
            [x.to(self.device, non_blocking=True) for x in x_list], dim=0
        )
    
        # Offset each graph's edge_index by its position in the batch.
        edge_index_batch = torch.cat(
            [
                edge_index_list[i].to(self.device, non_blocking=True) + i * num_nodes
                for i in range(num_graphs)
            ],
            dim=1
        )
    
        edge_attr_batch = None
        if edge_attr_list is not None and edge_attr_list[0] is not None:
            edge_attr_batch = torch.cat(
                [e.to(self.device, non_blocking=True) for e in edge_attr_list], dim=0
            )
    
        x_embed = self.node_embedding(x_batch)
    
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if edge_attr_batch is not None and self.model_type in ['gat', 'transformer']:
                h = torch.relu(bn(conv(x_embed, edge_index_batch, edge_attr_batch)))
            else:
                h = torch.relu(bn(conv(x_embed, edge_index_batch)))
            if weights is not None:
                h = weights[i] * h
            x_embed = x_embed + h
    
        # Split back into a list of per-graph embeddings so `forward()`
        # doesn't need to change at all.
        return list(torch.split(x_embed, num_nodes, dim=0))

    def forward(self, x_list, edge_index_list, edge_index_out_list,
                y_batch=None, mask=None, edge_attr_list=None):
        """
        Forward pass supporting both pretraining and finetuning.
        
        Args:
            x_list: Node feature tensors
            edge_index_list: Edge indices for message passing
            edge_index_out_list: Original graph edges for prediction
            y_batch: Labels (for finetuning) or target features (for pretraining)
            mask: Boolean mask indicating which values were masked
            
        Returns:
            For finetuning: edge predictions
            For pretraining: (reconstructed features, mask)
        """
        # Encode node features
        node_embeddings = self.encode(x_list, edge_index_list, edge_attr_list)
        
        if self.pretraining:
            # Reconstruction mode
            reconstructed = []
            for emb in node_embeddings:
                reconstructed.append(self.reconstruction_head(emb))
            return torch.cat(reconstructed, dim=0)
        else:
            # Edge classification mode
            all_edge_reprs = []
            for x_embed, orig_edges in zip(node_embeddings, edge_index_out_list):
                src, dst = orig_edges[0], orig_edges[1]
                all_edge_reprs.append(
                    torch.cat([x_embed[src], x_embed[dst]], dim=-1)
                )
            return self.fc(torch.cat(all_edge_reprs, dim=0))

class FeatureReconstructionHead(nn.Module):
    """
    Reconstruction head for masked pretraining.
    
    Takes node embeddings and predicts original features for masked nodes.
    Supports both continuous (MSE loss) and categorical (CrossEntropy loss) features.
    """
    
    def __init__(self, hidden_dim: int, output_dim: int, 
                 feature_types: List[str] = None):
        """
        Args:
            hidden_dim: Dimension of node embeddings
            output_dim: Number of features to reconstruct
            feature_types: List of 'continuous' or 'categorical' for each feature
        """
        super().__init__()
        
        self.feature_types = feature_types or ['continuous'] * output_dim
        
        # Main reconstruction network
        self.reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Separate heads for categorical features (if any)
        self.categorical_heads = nn.ModuleDict()
        for i, ftype in enumerate(self.feature_types):
            if ftype == 'categorical':
                # Assume categories are encoded as integers
                self.categorical_heads[str(i)] = nn.Linear(hidden_dim, 100)  # Adjust num_classes
        
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct features from node embeddings.
        
        Args:
            node_embeddings: Node embeddings from GNN encoder
            
        Returns:
            reconstructed_features: Predicted feature values
        """
        return self.reconstructor(node_embeddings)

class MaskedReconstructionLoss(nn.Module):
    """
    Combined loss for masked reconstruction.
    
    Handles both continuous (MSE) and categorical (CrossEntropy) features.
    """
    
    def __init__(self, feature_types: List[str], 
                 continuous_loss: str = 'mse',
                 categorical_weight: float = 1.0):
        """
        Args:
            feature_types: List of 'continuous' or 'categorical' for each feature
            continuous_loss: 'mse' or 'l1' for continuous features
            categorical_weight: Weight for categorical features in combined loss
        """
        super().__init__()
        self.feature_types = feature_types
        self.categorical_weight = categorical_weight
        
        if continuous_loss == 'mse':
            self.continuous_loss_fn = nn.MSELoss(reduction='none')
        elif continuous_loss == 'l1':
            self.continuous_loss_fn = nn.L1Loss(reduction='none')
        else:
            self.continuous_loss_fn = nn.MSELoss(reduction='none')
            
        self.categorical_loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss only on masked positions.
        
        Args:
            predictions: Reconstructed features
            targets: Original features
            mask: Boolean mask indicating masked positions
            
        Returns:
            Combined reconstruction loss
        """
        # Only compute loss on masked positions
        masked_preds = predictions[mask]
        masked_targets = targets[mask]
        
        if masked_preds.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        total_loss = 0.0
        feature_start = 0
        
        for i, ftype in enumerate(self.feature_types):
            if ftype == 'continuous':
                pred = masked_preds[:, i:i+1] if masked_preds.dim() > 1 else masked_preds
                target = masked_targets[:, i:i+1] if masked_targets.dim() > 1 else masked_targets
                total_loss += self.continuous_loss_fn(pred, target).mean()
            elif ftype == 'categorical':
                # For categorical features, use CrossEntropy
                pred = masked_preds[:, i]  # Assuming one-hot or class indices
                target = masked_targets[:, i].long()
                total_loss += self.categorical_weight * self.categorical_loss_fn(pred.unsqueeze(0), target.unsqueeze(0)).mean()
                
        return total_loss

# ============================================================================
# DATA GENERATOR
# ============================================================================

class MultiClassBatchGenerator(IterableDataset):
    """
    Iterable dataset that streams graph samples in chunks to reduce memory
    usage while supporting efficient GPU training.
    
    Now supports lazy loading: features are loaded on-demand from HDF5.
    """
    def __init__(self, feature_refs, neighbor_pairs, labels,
                 cells_array, mode="train", is_bi_directional=True,
                 batch_size=1, train_ratio=0.7, debug=False,
                 cluster_info_dict=None, chunk_size=2000, inference_only=False):
    
        # Store dataset configuration.
        self.debug = debug
        self.batch_size = batch_size
        self.mode = mode
        self.num_pairs = neighbor_pairs.shape[0]
        self.chunk_size = chunk_size
        self.inference_only = inference_only
        self.is_bi_directional = is_bi_directional
        self._cells_array = cells_array
    
        # Store references (lazy: features not loaded yet)
        self._feature_refs = feature_refs  # List of dicts with HDF5 references
        self._labels = labels  # Memory-mapped array
        self._cluster_info_dict = cluster_info_dict
    
        # Convert neighbor pairs to a PyTorch tensor.
        self.neighbor_pairs = torch.as_tensor(
            neighbor_pairs, dtype=torch.long
        )
    
        # ---- BUGFIX: Proper bidirectional edge handling ----
        raw_edges = self.neighbor_pairs.T.contiguous()
    
        if self.is_bi_directional:
            self.pairs_mp = to_undirected(raw_edges)
            self.pairs_pred = raw_edges
        else:
            self.pairs_mp = raw_edges
            self.pairs_pred = raw_edges
    
        # ---- SANITY CHECK: edge_attr (built in _compute_edge_features_from_array
        # as exactly 2 * pairs_pred.shape[1] rows) must line up with pairs_mp's
        # edge count, since GAT/Transformer pass both into the same conv call.
        # to_undirected() can deduplicate edges, which would silently break that
        # assumption on some datasets — fail fast at startup instead of mid-training.
        if self.is_bi_directional:
            expected_mp_edges = 2 * self.pairs_pred.shape[1]
            actual_mp_edges = self.pairs_mp.shape[1]
            if actual_mp_edges != expected_mp_edges:
                raise ValueError(
                    f"Edge count mismatch: pairs_mp has {actual_mp_edges} edges "
                    f"but edge_attr will be built with {expected_mp_edges} rows "
                    f"(2 * {self.pairs_pred.shape[1]} pairs_pred edges). "
                    f"to_undirected() likely deduplicated repeated pairs in "
                    f"neighbor_pairs. GAT/Transformer models will fail or "
                    f"silently misalign edge features until this is resolved "
                    f"(e.g. dedupe neighbor_pairs upstream, or drop edge_attr's "
                    f"reversed-copy construction to match to_undirected's output)."
                )
    
        # Pin memory to speed up CPU → GPU transfers.
        if torch.cuda.is_available():
            self.pairs_mp = self.pairs_mp.pin_memory()
            self.pairs_pred = self.pairs_pred.pin_memory()
    
        # Split events into training and validation/test sets.
        self.num_events = len(feature_refs)
        all_event_ids = list(range(self.num_events))
        split_idx = int(self.num_events * train_ratio)
    
        if mode == "train":
            self.event_indices = all_event_ids[:split_idx]
        else:
            self.event_indices = all_event_ids[split_idx:]
    
        pin_msg = " [pinned]" if torch.cuda.is_available() else ""
    
        log(
            f"📊 {mode.upper()} SET: {len(self.event_indices)} events "
            f"[CUDA, chunk={chunk_size}, bidirectional={is_bi_directional}{pin_msg}]"
        )
    
        self.num_chunks = (
            len(self.event_indices) + chunk_size - 1
        ) // chunk_size
    
        self._chunk_data = []

    def _compute_edge_features_from_array(self, features_np):
        """Same as before but takes numpy array directly instead of looking up in dict."""
        if self._cells_array is None:
            return None
        
        sin_phi = features_np[:, 5]
        cos_phi = features_np[:, 6]
        eta = features_np[:, 4]
        
        num_edges = self.pairs_pred.shape[1]
        src = self.pairs_pred[0, :num_edges].numpy()
        dst = self.pairs_pred[1, :num_edges].numpy()
        
        deta = eta[src] - eta[dst]
        dphi_sin = np.sin(np.arctan2(sin_phi[src], cos_phi[src]) - 
                         np.arctan2(sin_phi[dst], cos_phi[dst]))
        dphi_cos = np.cos(np.arctan2(sin_phi[src], cos_phi[src]) - 
                         np.arctan2(sin_phi[dst], cos_phi[dst]))
        dr = np.sqrt(deta**2 + np.arctan2(dphi_sin, dphi_cos)**2)
        
        if 'sampling' in self._cells_array.dtype.names:
            sampling = self._cells_array['sampling']
            same_layer = (sampling[src] == sampling[dst]).astype(np.float32)
        else:
            same_layer = np.ones(num_edges, dtype=np.float32)
        
        edge_attr = np.stack([
            deta.astype(np.float32), dphi_sin.astype(np.float32),
            dphi_cos.astype(np.float32), dr.astype(np.float32), same_layer
        ], axis=1)
        
        if self.is_bi_directional:
            edge_attr_rev = edge_attr.copy()
            edge_attr_rev[:, 0] = -edge_attr_rev[:, 0]
            edge_attr_rev[:, 1] = -edge_attr_rev[:, 1]
            edge_attr_full = np.concatenate([edge_attr, edge_attr_rev], axis=0)
        else:
            edge_attr_full = edge_attr
        
        return torch.from_numpy(edge_attr_full)
        
    def _load_event_features_from_open(self, ref, h5f):
        """
        Load a single event's features from an already-open HDF5 file.
        Avoids the overhead of opening/closing for every event.
        """
        local_idx = ref['local_idx']
        
        # Load SNR
        if 'cell/snr_computed' in h5f:
            snr_row = h5f['cell/snr_computed'][local_idx]
        elif 'cell/snr_raw' in h5f:
            snr_row = h5f['cell/snr_raw'][local_idx]
        elif 'cell/energy_raw' in h5f:
            energy = h5f['cell/energy_raw'][local_idx]
            noise = h5f['cell/noise_raw'][local_idx]
            noise_safe = np.where(noise == 0, 1e-6, noise)
            snr_row = energy / noise_safe
        else:
            snr_row = np.zeros(self._cells_array.shape[0], dtype=np.float32)
        
        # Load eta, phi
        if ref['has_eta']:
            eta_row = h5f['cell/cell_eta'][local_idx]
            phi_row = h5f['cell/cell_phi'][local_idx]
        else:
            eta_row = self._cells_array['eta_event0'].astype(np.float32)
            phi_row = self._cells_array['phi_event0'].astype(np.float32)
        
        # Apply feature engineering
        sin_phi = np.sin(phi_row).astype(np.float32)
        cos_phi = np.cos(phi_row).astype(np.float32)
        
        snr_f32 = snr_row.astype(np.float32)
        snr_scaled = np.sign(snr_f32) * np.log1p(np.abs(snr_f32))
        snr_gt4 = (np.abs(snr_f32) > 4).astype(np.float32)
        snr_gt2 = (np.abs(snr_f32) > 2).astype(np.float32)
        snr_gt0 = (np.abs(snr_f32) > 0).astype(np.float32)
        
        features = np.stack([
            snr_scaled, snr_gt4, snr_gt2, snr_gt0,
            eta_row, sin_phi, cos_phi
        ], axis=1).astype(np.float32)
        
        return features

    def _load_chunk_from_events(self, chunk_events, chunk_idx, num_chunks):
        """
        Load a chunk given an explicit list of event indices (so it works
        whether that list is the full split or a per-worker shard of it).
        """
        if self.debug or num_chunks <= 1 or chunk_idx % max(1, num_chunks // 5) == 0:
            log(
                f"  📂 Chunk {chunk_idx+1}/{num_chunks}: "
                f"events {chunk_events[0]}-{chunk_events[-1]} "
                f"({len(chunk_events)})"
            )
    
        chunk_samples = []
        current_hdf5_path = None
        current_h5f = None
    
        for event_idx in chunk_events:
            if event_idx >= len(self._feature_refs):
                continue
    
            ref = self._feature_refs[event_idx]
    
            if ref['hdf5_path'] != current_hdf5_path:
                if current_h5f is not None:
                    current_h5f.close()
                current_hdf5_path = ref['hdf5_path']
                current_h5f = h5py.File(current_hdf5_path, 'r')
    
            features = self._load_event_features_from_open(ref, current_h5f)
            edge_attr = self._compute_edge_features_from_array(features)
    
            x_scaled = torch.as_tensor(features, dtype=torch.float32)
            del features
    
            out_labels = torch.as_tensor(self._labels[event_idx].copy(), dtype=torch.long)
            if out_labels.dim() == 1:
                out_labels = out_labels.unsqueeze(1)
    
            cluster_info = (
                self._cluster_info_dict.get(event_idx)
                if self._cluster_info_dict
                else None
            )
    
            chunk_samples.append((
                x_scaled, self.pairs_mp, self.pairs_pred,
                out_labels, edge_attr, cluster_info, event_idx
            ))
    
        if current_h5f is not None:
            current_h5f.close()
    
        gc.collect()
        return chunk_samples
        
    def _free_chunk(self):
        """
        Release the currently loaded chunk to keep memory usage low.
        """
        if self._chunk_data:
            for sample in self._chunk_data:
                del sample
            del self._chunk_data
            self._chunk_data = []
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __iter__(self):
        """
        Iterate over the dataset one chunk at a time.
        Shards event_indices across DataLoader workers when num_workers > 0.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Give each worker a disjoint slice of the events.
            per_worker = int(np.ceil(len(self.event_indices) / worker_info.num_workers))
            w_start = worker_info.id * per_worker
            w_end = min(w_start + per_worker, len(self.event_indices))
            local_event_indices = self.event_indices[w_start:w_end]
        else:
            local_event_indices = self.event_indices
    
        local_num_chunks = (len(local_event_indices) + self.chunk_size - 1) // self.chunk_size
    
        for chunk_idx in range(local_num_chunks):
            self._free_chunk()
    
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, len(local_event_indices))
            chunk_events = local_event_indices[start:end]
    
            self._chunk_data = self._load_chunk_from_events(chunk_events, chunk_idx, local_num_chunks)
    
            for sample in self._chunk_data:
                yield sample
                if self.debug and chunk_idx == 0 and len(self._chunk_data[:5]) >= 5:
                    break
    
            if self.debug:
                break
    
        self._free_chunk()

    def __len__(self):
        """Return the number of events in the selected dataset split."""
        return len(self.event_indices)

    @staticmethod
    def collate_data(batch):
        """
        Custom DataLoader collation function.
        
        Batch elements are 7-tuples:
            (x, pairs_mp, pairs_pred, labels, edge_attr, cluster_info, event_idx)
        """
        return (
            [b[0] for b in batch],                    # 0: Node features
            [b[1] for b in batch],                    # 1: Message-passing edges
            [b[2] for b in batch],                    # 2: Output edges
            torch.cat([b[3] for b in batch], dim=0),  # 3: Edge labels
            [b[4] for b in batch] if batch[0][4] is not None else None,  # 4: Edge features
            [b[5] for b in batch],                    # 5: Cluster info (list of dicts or Nones)
            [b[6] for b in batch],                    # 6: Event indices
        )

# ============================================================================
# TRAINING AND PRE-TRAINING FUNCTIONS
# ============================================================================

def pretrain_epoch(model, loader, optimizer, criterion, masking_fn, 
                   scaler, device, feature_names, debug=False):
    """
    Pretrain one epoch with masked reconstruction.
    """
    model.train()
    total_loss = 0
    total_masked = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    # Determine if autocast should be enabled
    use_amp = scaler is not None
    
    for batch_idx, batch in enumerate(loader):
        x_list, ei_list, eio_list, _, edge_attr_list, _, event_ids = batch
        
        # Apply masking to each graph in the batch
        masked_x_list = []
        mask_list = []
        targets_list = []
        
        for i, x in enumerate(x_list):
            if masking_fn:
                event_id = event_ids[i] if event_ids else None
                masked_x, mask, targets = masking_fn.apply_mask(x, event_id=event_id, feature_names=feature_names)
                masked_x_list.append(masked_x)
                mask_list.append(mask)
                targets_list.append(targets)
            else:
                masked_x_list.append(x)
                
        # Move to device
        masked_x_list = [x.to(device, non_blocking=True) for x in masked_x_list]
        ei_list = [e.to(device, non_blocking=True) for e in ei_list]
        mask_list = [m.to(device, non_blocking=True) for m in mask_list]
        targets_list = [t.to(device, non_blocking=True) for t in targets_list]
        
        # ---- BUGFIX: Wrap forward pass in autocast for actual FP16 ----
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            predictions = model(masked_x_list, ei_list, eio_list, edge_attr_list=edge_attr_list)
            
            # ---- BUGFIX: predictions is one concatenated tensor covering
            # every graph in the batch (model.encode() batches them together).
            # Split it back per-graph by node count so the loss actually sees
            # every sample, instead of zip()'s previous silent truncation to
            # just the first graph in the batch.
            node_counts = [t.shape[0] for t in targets_list]
            pred_list = list(torch.split(predictions, node_counts, dim=0))
            
            loss = 0
            for pred, target, mask in zip(pred_list, targets_list, mask_list):
                loss += criterion(pred, target, mask)
            loss /= len(pred_list)
        
        # Backward pass with mixed precision
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.float().item()
        total_masked += sum(m.sum().item() for m in mask_list)
        
        if debug and batch_idx >= 2:
            break
    
    avg_loss = total_loss / max(1, len(loader))
    mask_ratio = total_masked / max(1, sum(m.numel() for m in mask_list))
    
    return {'loss': avg_loss, 'mask_ratio': mask_ratio}

def train_epoch(model, loader, optimizer, criterion, scaler,
                device, debug=False, accumulation_steps=1, epoch=0):
    """
    Train the model for one epoch.

    Supports mixed-precision training and gradient accumulation.
    """

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    # Reset gradients before starting the epoch.
    optimizer.zero_grad(set_to_none=True)

    # Determine if autocast should be enabled
    use_amp = scaler is not None

    for batch_idx, batch in enumerate(loader):

        x_list, ei_list, eio_list, y_batch, edge_attr_list, _, _ = batch

        # Move graph data and labels to GPU/target device.
        x_list = [x.to(device, non_blocking=True) for x in x_list]
        ei_list = [e.to(device, non_blocking=True) for e in ei_list]
        eio_list = [e.to(device, non_blocking=True) for e in eio_list]
        y_batch = y_batch.to(device, non_blocking=True).squeeze(1)

        # ---- BUGFIX: Wrap forward pass in autocast for actual FP16 ----
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            scores = model(x_list, ei_list, eio_list, edge_attr_list=edge_attr_list)
            # Scale the loss when using gradient accumulation.
            loss = criterion(scores, y_batch) / accumulation_steps

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer update
        if (batch_idx + 1) % accumulation_steps == 0:

            if scaler:
                # Unscale before stepping so gradients can be checked
                # or clipped correctly if needed.
                scaler.unscale_(optimizer)

                scaler.step(optimizer)
                scaler.update()

            else:
                optimizer.step()

            # Reset gradients for the next accumulation cycle.
            optimizer.zero_grad(set_to_none=True)

        # Multiply back by accumulation_steps because the loss was
        # divided earlier for gradient accumulation.
        # Use .float() to ensure loss is in FP32 for logging.
        total_loss += loss.float().item() * len(y_batch) * accumulation_steps

        # Convert logits into predicted class labels.
        # scores is FP16 when using autocast, but argmax works fine.
        preds = scores.argmax(dim=1)

        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return {
        "loss": total_loss / total if total else 0,
        "acc": correct / total if total else 0
    }

def compute_metrics_from_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_loss: float,
    total_samples: int,
    num_classes: int = 5
) -> Dict:
    """
    Compute classification metrics from NumPy arrays.

    Uses a vectorized confusion matrix implementation to efficiently
    compute per-class and aggregate metrics.
    """

    # Build the confusion matrix without explicit Python loops.
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)

    cm_sum = cm.sum()

    # Handle the edge case of an empty evaluation set.
    if cm_sum == 0:
        zero_dict = {
            "loss": 0,
            "accuracy": 0,
            "macro_f1": 0,
            "f1_sum_score": 0,
            "randomness_metric": 0,
            "macro_recall": 0,
            "macro_precision": 0,
            "weighted_recall": 0,
            "weighted_precision": 0,
            "weighted_f1": 0,
            "weighted_recall_score": 0,
            "weighted_f1_score": 0,
            "class_totals": [0] * num_classes,
            "confusion_matrix": cm.tolist()
        }

        zero_dict.update({
            f"recall_class_{c}": 0.0
            for c in range(num_classes)
        })

        zero_dict.update({
            f"precision_class_{c}": 0.0
            for c in range(num_classes)
        })

        zero_dict.update({
            f"f1_class_{c}": 0.0
            for c in range(num_classes)
        })

        return zero_dict

    # Compute true positives, false positives, and false negatives for
    # each class.
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP

    class_totals = cm.sum(axis=1)
    class_weights = class_totals / cm_sum

    # Compute per-class metrics.
    recall = np.divide(
        TP, TP + FN,
        out=np.zeros(num_classes),
        where=(TP + FN) > 0
    )

    precision = np.divide(
        TP, TP + FP,
        out=np.zeros(num_classes),
        where=(TP + FP) > 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes),
        where=(precision + recall) > 0
    )

    # Overall classification accuracy.
    accuracy = TP.sum() / cm_sum

    return {
        "loss": total_loss / total_samples if total_samples else 0,
        "accuracy": accuracy,

        # Per-class metrics.
        **{f"recall_class_{c}": recall[c] for c in range(num_classes)},
        **{f"precision_class_{c}": precision[c] for c in range(num_classes)},
        **{f"f1_class_{c}": f1[c] for c in range(num_classes)},

        # Macro-averaged metrics.
        "macro_recall": np.mean(recall),
        "macro_precision": np.mean(precision),
        "macro_f1": np.mean(f1),

        # Class-frequency-weighted metrics.
        "weighted_recall": np.sum(class_weights * recall),
        "weighted_precision": np.sum(class_weights * precision),
        "weighted_f1": np.sum(class_weights * f1),

        # Scaled summary scores used by this project.
        "randomness_metric": np.mean(recall) * num_classes,
        "f1_sum_score": np.mean(f1) * num_classes,
        "weighted_recall_score":
            np.sum(class_weights * recall) * num_classes,
        "weighted_f1_score":
            np.sum(class_weights * f1) * num_classes,

        # Additional diagnostic information.
        "class_totals": class_totals.tolist(),
        "confusion_matrix": cm.tolist(),
    }

def evaluate(model, loader, criterion, device,
             use_amp=False, num_classes=5):
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x_list, ei_list, eio_list, y_batch, edge_attr_list, _, _ in loader:  # ← FIXED

            x_list = [x.to(device, non_blocking=True) for x in x_list]
            ei_list = [e.to(device, non_blocking=True) for e in ei_list]
            eio_list = [e.to(device, non_blocking=True) for e in eio_list]
            y_batch = y_batch.to(device, non_blocking=True).squeeze(1)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                scores = model(x_list, ei_list, eio_list, edge_attr_list=edge_attr_list)

            total_loss += (
                criterion(scores.float(), y_batch).item()
                * len(y_batch)
            )

            preds = scores.argmax(dim=1)
            all_labels.append(y_batch.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            total += len(y_batch)

    y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    return compute_metrics_from_numpy(y_true, y_pred, total_loss, total, num_classes)

# ============================================================================
# BATCHED INFERENCE
# ============================================================================

@torch.no_grad()
def run_inference(model, generator, device, criterion=None, num_classes=5,
                  debug=False, show_progress=True, save_path=None,
                  model_name=None):
    """
    Run model inference over a dataset.

    Optionally computes loss and evaluation metrics during the same pass
    to avoid running a separate evaluation loop.

    Returns
    -------
    batch_results : list
        Predictions, probabilities, labels, and edge information.
    metrics_dict : dict or None
        Computed metrics if criterion is provided, otherwise None.
    """

    # Disable dropout/batch statistics updates and switch to evaluation mode.
    model.eval()

    # Store inference outputs before periodically writing them to disk.
    batch_results = []

    # Number of events accumulated before saving results.
    batch_size = 100

    total_events = len(generator)

    # Metric accumulators
    total_loss = 0.0
    total_samples = 0
    all_labels_list = []
    all_preds_list = []

    # Progress bar handling
    if show_progress and not debug:
        try:
            from tqdm import tqdm

            iterator = tqdm(
                enumerate(generator),
                total=total_events,
                desc="   🔮 Inference",
                unit="events"
            )

        except ImportError:
            # Fall back to a normal iterator if tqdm is unavailable.
            iterator = enumerate(generator)

    else:
        iterator = enumerate(generator)

    # Main inference loop
    for i, (x_scaled, edge_index, edge_index_out, y, edge_attr, cluster_info, _event_idx) in iterator:

        # Move graph data to the target device.
        x_scaled = x_scaled.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        edge_index_out = edge_index_out.to(device, non_blocking=True)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device, non_blocking=True)

        # Run the GNN.
        # Lists are used because the model supports batched graphs.
        out = model(
            [x_scaled],
            [edge_index],
            [edge_index_out],
            edge_attr_list=[edge_attr] if edge_attr is not None else None
        )

        # Convert logits into predictions and probabilities.
        preds = out.argmax(dim=1).cpu().numpy()

        scores = torch.softmax(
            out,
            dim=1
        ).cpu().numpy()

        # Extract source and destination nodes for each predicted edge.
        src_nodes = edge_index_out[0].cpu().numpy()
        dst_nodes = edge_index_out[1].cpu().numpy()

        # Convert labels into NumPy format for metric computation.
        labels_np = (
            y.squeeze(1).numpy()
            if y is not None and y.dim() == 2
            else (y.numpy() if y is not None else None)
        )

        # Inline metric accumulation
        if criterion is not None and labels_np is not None:

            y_tensor = y.to(
                device,
                non_blocking=True
            ).squeeze(1)

            loss_val = (
                criterion(out, y_tensor)
                .float()
                .item()
                * len(y_tensor)
            )

            total_loss += loss_val
            total_samples += len(y_tensor)

            all_labels_list.append(labels_np)
            all_preds_list.append(preds)

        # Store inference results for this event.
        batch_results.append({
            "event_id": (
                generator.event_indices[i]
                if hasattr(generator, 'event_indices')
                else i
            ),
            "preds": preds,
            "scores": scores,
            "labels": labels_np,

            # Original edge pairs used for prediction.
            "neighbor_pairs": np.stack(
                [src_nodes, dst_nodes],
                axis=1
            ),

            # Optional cluster metadata.
            "cluster_info": cluster_info,
        })

        # Periodic saving
        if len(batch_results) >= batch_size and save_path:

            log(
                f"  💾 Saving batch ({i+1}/{total_events})..."
            )

            save_results_to_parquet(
                batch_results,
                save_path,
                model_name,
                append=True
            )

            batch_results = []

            # Release unused Python/GPU memory.
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Debug mode: only process a few events.
        if debug and i >= 4:
            log("Debug: stopping after 5 events")
            break

    # Save any remaining inference results.
    if batch_results and save_path:
        save_results_to_parquet(
            batch_results,
            save_path,
            model_name,
            append=True
        )

    # Final metric computation
    metrics_dict = None

    if criterion is not None and all_labels_list:

        # Combine predictions from all events into one array.
        y_true = np.concatenate(all_labels_list)
        y_pred = np.concatenate(all_preds_list)

        # Compute confusion matrix and derived metrics.
        metrics_dict = compute_metrics_from_numpy(
            y_true,
            y_pred,
            total_loss,
            total_samples,
            num_classes
        )

    return batch_results, metrics_dict


# ============================================================================
# PARQUET SAVING
# ============================================================================

def save_results_to_parquet(results, save_path, model_name,
                            cluster_info_dict=None, append=False):
    """
    Convert inference outputs into a flat edge-level Parquet table.

    Each row corresponds to one predicted edge and contains:
        - event ID
        - source/destination node IDs
        - true and predicted labels
        - prediction confidence
        - class probabilities
        - optional cluster information

    Results are written incrementally to avoid large memory usage.
    """

    # Nothing to save.
    if not results:
        return ""

    # Remove previous output when starting a fresh write.
    if not append and os.path.exists(save_path):
        os.remove(save_path)

    if not append:
        log("   💾 Saving to Parquet...")

    writer = None

    # When appending, preserve the original Parquet schema.
    # This ensures all batches have identical column types.
    if append and os.path.exists(save_path):
        existing_schema = pq.ParquetFile(save_path).schema_arrow

    # Check whether cluster information is available.
    has_cluster = bool(
        results and results[0].get('cluster_info')
    )

    # Process results in small chunks to control memory usage.
    for chunk_start in range(0, len(results), 10):

        chunk_end = min(chunk_start + 10, len(results))
        chunk_results = results[chunk_start:chunk_end]

        # Total number of edges across this chunk.
        chunk_edges = sum(
            len(r['preds'])
            for r in chunk_results
        )

        # Allocate output arrays.
        event_ids = np.zeros(chunk_edges, dtype=np.int32)
        edge_ids = np.zeros(chunk_edges, dtype=np.int32)

        src = np.zeros(chunk_edges, dtype=np.int32)
        dst = np.zeros(chunk_edges, dtype=np.int32)

        true_labels = np.full(
            chunk_edges,
            -1,
            dtype=np.int8
        )

        pred_labels = np.zeros(
            chunk_edges,
            dtype=np.int8
        )

        confidence = np.zeros(
            chunk_edges,
            dtype=np.float32
        )

        # Store probability for each class.
        scores = np.zeros(
            (chunk_edges, 5),
            dtype=np.float32
        )

        # Optional cluster IDs.
        if has_cluster:
            src_cluster = np.full(
                chunk_edges,
                -1,
                dtype=np.int32
            )
            dst_cluster = np.full(
                chunk_edges,
                -1,
                dtype=np.int32
            )

        offset = 0

        # Flatten event-level predictions into edge-level rows.
        for result in chunk_results:

            n = len(result['preds'])

            event_ids[offset:offset+n] = result['event_id']
            edge_ids[offset:offset+n] = np.arange(n)

            src[offset:offset+n] = result['neighbor_pairs'][:, 0]
            dst[offset:offset+n] = result['neighbor_pairs'][:, 1]

            if result['labels'] is not None:
                true_labels[offset:offset+n] = result['labels']

            pred_labels[offset:offset+n] = result['preds']

            # Extract probability assigned to the predicted class.
            confidence[offset:offset+n] = (
                result['scores'][
                    np.arange(n),
                    result['preds']
                ]
            )

            scores[offset:offset+n] = result['scores']

            # Add cluster-level information if available.
            if has_cluster:
                cidx = result['cluster_info'].get(
                    'cell_cluster_index'
                )

                if cidx is not None:
                    src_cluster[offset:offset+n] = (
                        cidx[result['neighbor_pairs'][:, 0]]
                    )

                    dst_cluster[offset:offset+n] = (
                        cidx[result['neighbor_pairs'][:, 1]]
                    )

            offset += n

        # Create a tabular representation for Parquet storage.
        df = pd.DataFrame({
            'event_id': event_ids,
            'edge_id': edge_ids,
            'source_id': src,
            'target_id': dst,
            'true_label': true_labels,
            'pred_label': pred_labels,
            'confidence': confidence,
            'score_class_0': scores[:, 0],
            'score_class_1': scores[:, 1],
            'score_class_2': scores[:, 2],
            'score_class_3': scores[:, 3],
            'score_class_4': scores[:, 4],
            'model_name': model_name
        })

        if has_cluster:
            df['source_cluster'] = src_cluster
            df['target_cluster'] = dst_cluster

            # Useful diagnostic:
            # whether both nodes belong to the same reconstructed cluster.
            df['same_cluster'] = (
                (src_cluster == dst_cluster)
                & (src_cluster >= 0)
            )

        table = pa.Table.from_pandas(
            df,
            preserve_index=False
        )

        # Create the Parquet writer only once.
        if writer is None:
            schema = (
                existing_schema
                if (append and os.path.exists(save_path))
                else table.schema
            )

            writer = pq.ParquetWriter(
                save_path,
                schema,
                compression='zstd',
                compression_level=3,
                use_dictionary=True,
                write_statistics=True
            )

        writer.write_table(table)

        # Release temporary memory.
        del df, table
        gc.collect()

    if writer:
        writer.close()

    if not append:
        log(
            f"   💾 Saved: {save_path} "
            f"({os.path.getsize(save_path)/1024**3:.2f} GB)"
        )

    return save_path

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model_full(model, train_loader, test_loader, test_generator, optimizer, criterion, scaler,
                     device, args, model_name, cluster_info=None, tracker=None, scheduler=None):
    """
    Complete training pipeline.

    Handles:
        - checkpoint loading/resume
        - epoch training
        - evaluation
        - best-model selection
        - early stopping
        - final inference
        - metric saving
    """
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, model_name)
    best_model_path = os.path.join(args.save_dir, f"best_{model_name}")
    metrics_path = os.path.splitext(model_path)[0] + "_metrics.pkl"
    
    best_f1_sum_score = 0.0; best_epoch = 0; start_epoch = 1
    metrics = {"train_loss":[],"test_loss":[],"train_accuracy":[],"test_accuracy":[],
               "test_macro_recall":[],"test_macro_precision":[],"test_macro_f1":[],
               "test_weighted_recall":[],"test_weighted_precision":[],"test_weighted_f1":[],
               "test_randomness_metric":[],"test_f1_sum_score":[],"test_weighted_recall_score":[],"test_weighted_f1_score":[],
               "epoch_times":[],
               "best_accuracy":0.0,"best_macro_recall":0.0,"best_macro_precision":0.0,"best_macro_f1":0.0,
               "best_weighted_recall":0.0,"best_weighted_precision":0.0,"best_weighted_f1":0.0,
               "best_randomness_metric":0.0,"best_f1_sum_score":0.0,
               "best_weighted_recall_score":0.0,"best_weighted_f1_score":0.0,
               "best_epoch":0,"total_time":0.0,"args":vars(args)}
    for c in range(5):
        for m in ['recall','precision','f1']: metrics[f"test_{m}_class_{c}"] = []

    if args.resume:
        chk = find_latest_checkpoint(args.save_dir, model_name)
        if chk:
            ckpt = torch.load(chk[1], map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict']); optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scaler and 'scaler_state_dict' in ckpt: scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = chk[0]+1; log(f"[Resume] Loaded checkpoint: {chk[1]}")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path,'rb') as f: metrics.update(pickle.load(f))
                    best_f1_sum_score = metrics.get("best_f1_sum_score",0.0); best_epoch = metrics.get("best_epoch",0)
                except: pass
    
    early_counter = 0; log(f"\n🚀 Starting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs+1):
        t0 = time.perf_counter()
        train_res = train_epoch(model, train_loader, optimizer, criterion, scaler, device, args.debug, epoch=epoch)
        if scheduler is not None:
            scheduler.step()
        test_res = evaluate(model, test_loader, criterion, device)
        dt = time.perf_counter()-t0
        
        metrics["epoch_times"].append(dt); metrics["train_loss"].append(train_res["loss"]); metrics["train_accuracy"].append(train_res["acc"])
        metrics["test_loss"].append(test_res["loss"]); metrics["test_accuracy"].append(test_res["accuracy"])
        metrics["test_macro_recall"].append(test_res["macro_recall"]); metrics["test_macro_precision"].append(test_res["macro_precision"])
        metrics["test_macro_f1"].append(test_res["macro_f1"])
        metrics["test_weighted_recall"].append(test_res["weighted_recall"]); metrics["test_weighted_precision"].append(test_res["weighted_precision"])
        metrics["test_weighted_f1"].append(test_res["weighted_f1"])
        metrics["test_randomness_metric"].append(test_res["randomness_metric"]); metrics["test_f1_sum_score"].append(test_res["f1_sum_score"])
        metrics["test_weighted_recall_score"].append(test_res["weighted_recall_score"]); metrics["test_weighted_f1_score"].append(test_res["weighted_f1_score"])
        for c in range(5):
            for m in ['recall','precision','f1']: metrics[f"test_{m}_class_{c}"].append(test_res.get(f'{m}_class_{c}',0.0))
        
        if test_res["f1_sum_score"] > best_f1_sum_score+0.01:
            best_f1_sum_score = test_res["f1_sum_score"]; best_epoch = epoch
            for key in ["accuracy","macro_recall","macro_precision","macro_f1","weighted_recall","weighted_precision","weighted_f1",
                       "randomness_metric","f1_sum_score","weighted_recall_score","weighted_f1_score"]:
                metrics[f"best_{key}"] = test_res[key]
            metrics["best_epoch"] = best_epoch; early_counter = 0
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),
                        **({"scaler_state_dict":scaler.state_dict()} if scaler else {})}, best_model_path)
            log(f"  💾 Best model (F1_Sum={best_f1_sum_score:.2f})")
        else: early_counter += 1
        
        if not args.debug:
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()},
                      os.path.join(args.save_dir, f"{os.path.splitext(model_name)[0]}_epoch{epoch}.pt"))
        
        if args.debug or epoch%5==0: log(f"[Epoch {epoch}] {dt:.1f}s F1_Sum={test_res['f1_sum_score']:.2f} Best={best_f1_sum_score:.2f}")
        if tracker: tracker.log_measurement(f"epoch_{epoch}_complete", f"F1_Sum={test_res['f1_sum_score']:.2f}")
        if early_counter >= args.patience: log(f"[Early Stop] epoch {epoch}"); break
    
    metrics["total_time"] = sum(metrics["epoch_times"])
    
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model_base = os.path.splitext(model_name)[0]
        parquet_path = os.path.join(args.save_dir, f"results_{model_base}.parquet")
        
        _, final_metrics = run_inference(
            model, test_generator, device,
            criterion=nn.CrossEntropyLoss(),
            num_classes=5,
            debug=args.debug, show_progress=True,
            save_path=parquet_path, model_name=model_base
        )

        if final_metrics and final_metrics.get("accuracy", 0) > 0 and sum(final_metrics.get("class_totals", [])) > 0:
            metrics["final_inference_metrics"] = final_metrics  # keep as a separate, clearly-labeled record
        else:
            log("⚠️ Final inference pass produced no labeled samples — keeping training-loop best_* metrics unchanged")
        
        metrics["num_events_evaluated"] = len(test_generator); metrics["parquet_results_path"] = parquet_path
        save_pickle(metrics, metrics_path); log(f"📊 Metrics saved: {metrics_path}")
    
    return metrics, model, best_model_path


# ============================================================================
# SINGLE MODEL TRAINING (UPDATED WITH PRETRAINING SUPPORT)
# ============================================================================

def train_single_model(args, model_type=None, tracker=None):
    model_name_used = model_type or args.model

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(args.gpu)
        log(f"🎯 GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu"); log("🎯 CPU")
    if tracker: tracker.log_measurement("device_set")
    
    feature_refs, pairs, labels, cluster_info, input_dim, feature_names, cells = load_features_lazy(args.data_dir, args, tracker)
    
    exp_name = args.exp_name or f"{model_name_used}_{'baseline' if not args.all_features else 'all'}_h{args.hidden_dim}_l{args.layers}"
    if model_name_used in ['gat','transformer']: exp_name += f"_heads{args.heads}"
    if args.weighted_loss: exp_name += f"_{args.weight_strategy}"
    if args.pretrain: exp_name += f"_pretrain_{args.mask_type}_r{args.mask_ratio}"
    model_filename = f"{exp_name}.pt"
    
    log(f"\n{'='*60}\n🔬 EXPERIMENT: {exp_name}\n{'='*60}")
    log(f"   Model: {model_name_used.upper()} | Features: {input_dim} | Hidden: {args.hidden_dim} | Layers: {args.layers}")
    if args.pretrain:
        log(f"   Pretraining: {args.mask_type} masking | Ratio: {args.mask_ratio} | Pretrain epochs: {args.pretrain_epochs}")
    
    # ---- INFERENCE-ONLY MODE ----
    if args.inference_only:
        log(f"\n{'='*60}\n🔮 INFERENCE-ONLY MODE\n{'='*60}")
        best_model_path = os.path.join(args.save_dir, f"best_{model_filename}")
        if not os.path.exists(best_model_path):
            chk = find_latest_checkpoint(args.save_dir, model_filename)
            if chk: best_model_path = chk[1]
            else: log("❌ No checkpoint found."); return None, None
        
        ckpt = torch.load(best_model_path, map_location=device, weights_only=True)
        
        # Check if this is a pretrained model (has reconstruction head)
        is_pretrained = 'reconstruction_head.reconstructor.0.weight' in ckpt.get('model_state_dict', {})
        
        # Create model in finetuning mode for inference
        model = GraphFoundationModel(input_dim, args.hidden_dim, 5, device, model_name_used, args.layers,
                                     args.heads, args.dropout, args.layer_weights, args.softmax_weights, 
                                     args.norm, args.debug, pretraining=False, feature_names=feature_names).to(device)
        
        # Load only matching parameters
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items() 
                          if k in model_dict and 'reconstruction_head' not in k and 'fc' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()
        
        model_base = os.path.splitext(model_filename)[0]
        parquet_path = os.path.join(args.save_dir, f"results_{model_base}.parquet")
        
        test_generator = MultiClassBatchGenerator(
            feature_refs, pairs, labels, cells, mode='test',
            cluster_info_dict=cluster_info,
            debug=args.debug, is_bi_directional=True,
            train_ratio=args.train_ratio,
            chunk_size=2000, inference_only=False
        )
        
        _, test_metrics = run_inference(
            model, test_generator, device,
            criterion=nn.CrossEntropyLoss(),
            num_classes=5,
            debug=args.debug, show_progress=True,
            save_path=parquet_path, model_name=model_base
        )
        
        if test_metrics is None:
            log("❌ Inference produced no metrics")
            return None, None
        
        metrics = {'best_epoch': ckpt.get('epoch', '?'),
                   'best_accuracy': test_metrics['accuracy'],
                   'best_macro_f1': test_metrics['macro_f1'],
                   'best_f1_sum_score': test_metrics['f1_sum_score'],
                   'best_randomness_metric': test_metrics['randomness_metric'],
                   'parquet_results_path': parquet_path,
                   'model_path': best_model_path,
                   'args': vars(args)}
        for c in range(5):
            for m in ['recall', 'precision', 'f1']:
                metrics[f'test_{m}_class_{c}'] = [test_metrics.get(f'{m}_class_{c}', 0.0)]
        save_pickle(metrics, os.path.join(args.save_dir, f"{model_base}_metrics.pkl"))
        log(f"\n✅ INFERENCE COMPLETE! F1_Sum: {test_metrics['f1_sum_score']:.2f} | Parquet: {parquet_path}")
        return metrics, best_model_path
    
    # ---- PRETRAINING MODE ----
    if args.pretrain:
        log(f"\n{'='*60}\n🔧 PRETRAINING MODE\n{'='*60}")
        
        # Create masking function
        masking_fn = CalorimeterMasking(
            mask_ratio=args.mask_ratio,
            mask_type=args.mask_type,
            mask_features=args.mask_features,
            cells_array=cells,
            cluster_info_dict=cluster_info,
            geometry_radius=args.geometry_radius
        )
        
        # Create pretraining model
        pretrain_model = GraphFoundationModel(
            input_dim, args.hidden_dim, 5, device,
            model_type=model_name_used,
            num_layers=args.layers,
            num_heads=args.heads,
            dropout=args.dropout,
            layer_weights=args.layer_weights,
            softmax_weights=args.softmax_weights,
            norm_type=args.norm,
            debug=args.debug,
            pretraining=True,
            feature_names=feature_names
        ).to(device)
        
        log(f"   Pretrain params: {sum(p.numel() for p in pretrain_model.parameters() if p.requires_grad):,}")
        
        # Create reconstruction loss
        recon_criterion = MaskedReconstructionLoss(
            feature_types=pretrain_model._infer_feature_types(),
            continuous_loss=args.continuous_loss
        )
        
        # Create optimizer and scaler for pretraining
        pretrain_optimizer = optim.Adam(pretrain_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        pretrain_scaler = torch.amp.GradScaler('cuda') if (args.mixed_precision and args.gpu>=0 and torch.cuda.is_available()) else None
        log(f"✅ Pretraining: {'FP16' if pretrain_scaler else 'FP32'}")

        # ---- RESTORED: dataset/loader creation for pretraining ----
        pretrain_generator = MultiClassBatchGenerator(
            feature_refs, pairs, labels, cells,
            mode='train',
            cluster_info_dict=cluster_info,
            debug=args.debug, is_bi_directional=True,
            train_ratio=1.0,  # Use all data for pretraining
            chunk_size=2000, inference_only=False
        )

        pretrain_loader = DataLoader(pretrain_generator, batch_size=args.batch_size,
                                    collate_fn=MultiClassBatchGenerator.collate_data, 
                                    pin_memory=True, num_workers=0)

        pretrained_path = os.path.join(args.save_dir, f"pretrained_{model_filename}")
        pretrain_metrics_path = os.path.join(args.save_dir, f"pretrain_metrics_{exp_name}.pkl")

        # ---- RESUME SUPPORT ----
        # Unlike train_model_full(), pretraining previously had no resume
        # logic: every restart began at epoch 1 with best_pretrain_loss
        # reset to inf, silently overwriting any existing checkpoint on the
        # very first epoch. This mirrors the checkpoint/resume pattern
        # already used for finetuning.
        start_pretrain_epoch = 1
        best_pretrain_loss = float('inf')
        pretrain_metrics_history = []

        if args.resume and os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location=device, weights_only=True)
            pretrain_model.load_state_dict(ckpt['model_state_dict'])
            pretrain_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if pretrain_scaler and 'scaler_state_dict' in ckpt:
                pretrain_scaler.load_state_dict(ckpt['scaler_state_dict'])

            start_pretrain_epoch = ckpt.get('epoch', 0) + 1
            best_pretrain_loss = ckpt.get('pretrain_loss', float('inf'))

            log(
                f"[Resume] Loaded pretrained checkpoint: {pretrained_path} "
                f"(epoch {ckpt.get('epoch', '?')}, loss={best_pretrain_loss:.6f})"
            )

            if os.path.exists(pretrain_metrics_path):
                try:
                    pretrain_metrics_history = load_pickle(pretrain_metrics_path)
                    log(f"[Resume] Loaded {len(pretrain_metrics_history)} prior pretrain epoch records")
                except Exception as e:
                    log(f"⚠️ Could not load pretrain metrics history, starting fresh: {e}")
                    pretrain_metrics_history = []

            if start_pretrain_epoch > args.pretrain_epochs:
                log(
                    f"✅ Pretraining already complete "
                    f"({start_pretrain_epoch - 1}/{args.pretrain_epochs} epochs done) — skipping ahead"
                )

        # Pretraining loop
        log(f"\n🚀 Starting pretraining from epoch {start_pretrain_epoch} to {args.pretrain_epochs}...")

        for epoch in range(start_pretrain_epoch, args.pretrain_epochs + 1):
            t0 = time.perf_counter()
            
            pretrain_res = pretrain_epoch(
                pretrain_model, pretrain_loader, pretrain_optimizer, 
                recon_criterion, masking_fn, pretrain_scaler, device, 
                feature_names, args.debug
            )
            
            dt = time.perf_counter() - t0
            pretrain_metrics_history.append({
                'epoch': epoch,
                'loss': pretrain_res['loss'],
                'mask_ratio': pretrain_res['mask_ratio'],
                'time': dt
            })
            
            # Save best pretrained model
            if pretrain_res['loss'] < best_pretrain_loss:
                best_pretrain_loss = pretrain_res['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': pretrain_model.state_dict(),
                    'optimizer_state_dict': pretrain_optimizer.state_dict(),
                    **({'scaler_state_dict': pretrain_scaler.state_dict()} if pretrain_scaler else {}),
                    'feature_names': feature_names,
                    'input_dim': input_dim,
                    'hidden_dim': args.hidden_dim,
                    'mask_type': args.mask_type,
                    'mask_ratio': args.mask_ratio,
                    'pretrain_loss': pretrain_res['loss'],
                }, pretrained_path)
            
            if epoch == 1 or epoch % max(1, args.pretrain_epochs // 10) == 0 or epoch == args.pretrain_epochs:
                log(f"[Pretrain Epoch {epoch}/{args.pretrain_epochs}] {dt:.1f}s | Loss: {pretrain_res['loss']:.6f} | Mask ratio: {pretrain_res['mask_ratio']:.3f}")
            
            if tracker: 
                tracker.log_measurement(f"pretrain_epoch_{epoch}", 
                                       f"Loss={pretrain_res['loss']:.4f}")
        
        # Save pretraining metrics
        # ---- CLEANUP: pretrain_metrics_path was previously recomputed
        # here a second time (identical value, just redundant). It's
        # already set above where it's needed for the resume-loading step.
        save_pickle(pretrain_metrics_history, pretrain_metrics_path)
        log(f"💾 Pretraining complete! Best loss: {best_pretrain_loss:.6f}")
        
        if not args.finetune_epochs or args.finetune_epochs == 0:
            log("✅ Pretraining only mode - skipping finetuning")
            if tracker: 
                tracker.log_measurement("pretraining_complete")
                tracker.print_summary()
                tracker.save_report(f"resource_report_{exp_name}.json")
            return pretrain_metrics_history, pretrained_path
        
        # ---- TRANSFER TO FINETUNING MODEL ----
        log(f"\n{'='*60}\n🔧 FINETUNING MODE (from pretrained encoder)\n{'='*60}")
        
        # Create finetuning model
        model = GraphFoundationModel(
            input_dim, args.hidden_dim, 5, device,
            model_type=model_name_used,
            num_layers=args.layers,
            num_heads=args.heads,
            dropout=args.dropout,
            layer_weights=args.layer_weights,
            softmax_weights=args.softmax_weights,
            norm_type=args.norm,
            debug=args.debug,
            pretraining=False,  # Use classification head
            feature_names=feature_names
        ).to(device)
        
        # Transfer pretrained encoder weights
        pretrained_dict = pretrain_model.state_dict()
        finetune_dict = model.state_dict()
        
        # Only copy encoder weights (not task-specific heads)
        transferred_count = 0
        for k, v in pretrained_dict.items():
            if k in finetune_dict and 'reconstruction_head' not in k:
                finetune_dict[k] = v
                transferred_count += 1
        
        model.load_state_dict(finetune_dict)
        log(f"✅ Transferred {transferred_count} parameters from pretrained encoder")
        
        # Use finetuning epochs
        args.epochs = args.finetune_epochs
        
    else:
        # ---- NORMAL TRAINING MODE (NO PRETRAINING) ----
        # Use training labels for class weight computation
        model = GraphFoundationModel(input_dim, args.hidden_dim, 5, device, model_name_used, args.layers,
                                     args.heads, args.dropout, args.layer_weights, args.softmax_weights, 
                                     args.norm, args.debug, pretraining=False, feature_names=feature_names).to(device)
        log(f"   Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        if tracker: tracker.log_measurement("model_created")
    
    # ---- COMMON TRAINING SETUP ----
    # Reuse the generator's event list
    all_event_ids = list(range(len(feature_refs)))
    split_idx = int(len(all_event_ids) * args.train_ratio)
    
    # Extract only training labels to avoid data leakage
    train_labels = labels[all_event_ids[:split_idx]].flatten()

    criterion = create_loss_function(args, train_labels, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # ---- BUGFIX #9: Cosine annealing LR with warmup ----
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_epochs = min(5, max(1, args.epochs // 6))
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    
    scaler = torch.amp.GradScaler('cuda') if (args.mixed_precision and args.gpu>=0 and torch.cuda.is_available()) else None
    log(f"✅ {'FP16' if scaler else 'FP32'} | LR: cosine+{warmup_epochs}ep warmup")
    
    train_generator = MultiClassBatchGenerator(
        feature_refs, pairs, labels, cells,
        mode='train',
        cluster_info_dict=cluster_info,
        debug=args.debug, is_bi_directional=True,
        train_ratio=args.train_ratio,
        chunk_size=2000, inference_only=False
    )
    test_generator = MultiClassBatchGenerator(
        feature_refs, pairs, labels, cells,
        mode='test',
        cluster_info_dict=cluster_info,
        debug=args.debug, is_bi_directional=True,
        train_ratio=args.train_ratio,
        chunk_size=2000, inference_only=False
    )
    train_loader = DataLoader(
        train_generator, batch_size=args.batch_size,
        collate_fn=MultiClassBatchGenerator.collate_data,
        pin_memory=True, num_workers=4, persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_generator, batch_size=args.batch_size,
        collate_fn=MultiClassBatchGenerator.collate_data,
        pin_memory=True, num_workers=2, persistent_workers=True,
        prefetch_factor=2
    )
    if tracker: tracker.log_measurement("data_loaders_ready")
    
    metrics, model, model_path = train_model_full(model, train_loader, test_loader, test_generator,
                                                   optimizer, criterion, scaler, device, args, model_filename,
                                                   cluster_info, tracker, scheduler=scheduler)
    
    log(f"\n{'='*60}\n🏁 TRAINING SUMMARY\n{'='*60}")
    log(f"   Best epoch: {metrics['best_epoch']} | F1_Sum: {metrics['best_f1_sum_score']:.2f} | Accuracy: {metrics['best_accuracy']:.4f}")
    log(f"   Total time: {metrics['total_time']/60:.1f} min")
    if tracker: tracker.log_measurement("training_complete"); tracker.print_summary(); tracker.save_report(f"resource_report_{exp_name}.json")
    return metrics, model_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    log(f"{'='*70}\n🚀 GRAPH FOUNDATION MODEL (CUDA)\n{'='*70}")
    log(f"Model: {args.model.upper()} | Features: {'ALL (42+)' if args.all_features else 'BASELINE (3)'}")
    log(f"Hidden: {args.hidden_dim} | Layers: {args.layers} | GPU: {args.gpu}")
    
    if args.pretrain:
        log(f"Mode: PRETRAINING + {'FINETUNING' if args.finetune_epochs > 0 else 'PRETRAINING ONLY'}")
        log(f"Mask: {args.mask_type} (ratio={args.mask_ratio})")
    elif args.inference_only:
        log(f"Mode: INFERENCE ONLY")
    else:
        log(f"Mode: SUPERVISED TRAINING")
    
    tracker = ResourceTracker(log_dir=args.save_dir, enabled=args.track_resources)
    if args.track_resources: tracker.start()
    
    if args.analyze_scalability:
        analyze_dataset_scalability(args.data_dir)
        if tracker: tracker.save_report("scalability_report.json")
        return
    
    if args.model == 'all':
        for arch in ['gcn','gat','transformer','sage']:
            log(f"\n{'='*60}\nTraining {arch.upper()}...")
            try: 
                # Create a copy of args with the specific architecture
                import copy
                arch_args = copy.deepcopy(args)
                arch_args.model = arch
                train_single_model(arch_args, arch, tracker)
            except Exception as e: 
                log(f"❌ {arch} failed: {e}"); traceback.print_exc()
            torch.cuda.empty_cache(); gc.collect()
    else:
        try: 
            train_single_model(args, tracker=tracker)
        except Exception as e: 
            log(f"❌ Failed: {e}"); traceback.print_exc(); sys.exit(1)
    
    if tracker: 
        tracker.print_summary()
        tracker.save_report("final_resource_report.json")
    
    log(f"\n{'='*70}\n🎉 PIPELINE COMPLETE!\n{'='*70}")

if __name__ == "__main__":
    main()