#!/usr/bin/env python3
"""
Graph Foundation Model for Particle Physics Edge Classification

CUDA/LINUX VERSION - FULLY INTEGRATED & OPTIMIZED

Three paradigms:
  - edge_classification : per-edge 5-class boundary prediction
  - embedding           : supervised contrastive node embeddings + clusterer
  - cluster_slots       : end-to-end cluster assignment via Slot Attention
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
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MPLBACKEND", "Agg")
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

try:
    from hierarchical_split_reattach import (
        hierarchical_split, build_clusters_from_mask
    )
    SPLIT_AVAILABLE = True
except ImportError:
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from hierarchical_split_reattach import (
            hierarchical_split, build_clusters_from_mask
        )
        SPLIT_AVAILABLE = True
    except ImportError:
        SPLIT_AVAILABLE = False
        hierarchical_split = None
        build_clusters_from_mask = None


# ============================================================================
# LOGGING / GLOBAL SETUP
# ============================================================================

def log(msg: str) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


if mp.get_start_method(allow_none=True) != 'spawn':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

torch.backends.cudnn.benchmark = True
torch.set_num_threads(2)


# ============================================================================
# RESOURCE TRACKER
# ============================================================================

class ResourceTracker:
    """Tracks runtime, memory, CPU, and optional GPU usage during execution."""

    def __init__(self, log_dir: str = None, enabled: bool = True):
        self.enabled = enabled
        self.log_dir = log_dir or './resource_logs'
        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)
        self.measurements = []
        self.process = psutil.Process() if self.enabled else None
        self.start_time = None
        self.start_memory = None
        self.start_gpu_memory = None

    def start(self):
        if not self.enabled:
            return
        self.start_time = time.perf_counter()
        self.start_memory = self.process.memory_info().rss / 1024**3
        if torch.cuda.is_available():
            self.start_gpu_memory = torch.cuda.memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()

    def measure(self, stage: str) -> Optional[Dict]:
        if not self.enabled:
            return None
        mem = self.process.memory_info().rss / 1024**3
        elapsed = time.perf_counter() - self.start_time
        measurement = {
            'stage': stage,
            'timestamp': datetime.datetime.now().isoformat(),
            'time_elapsed': elapsed,
            'memory_gb': mem,
            'memory_delta_gb': mem - self.start_memory,
        }
        # Report peak GPU usage since last reset, not instantaneous live
        # tensors (between epochs, live-tensor count collapses even after
        # heavy work).
        if torch.cuda.is_available():
            measurement['gpu_memory_current_gb'] = torch.cuda.memory_allocated() / 1024**3
            measurement['gpu_memory_peak_gb'] = torch.cuda.max_memory_allocated() / 1024**3
            measurement['gpu_memory_reserved_peak_gb'] = torch.cuda.max_memory_reserved() / 1024**3
            measurement['gpu_memory_delta_gb'] = (
                measurement['gpu_memory_current_gb'] - self.start_gpu_memory
            )
            measurement['gpu_memory_type'] = 'cuda'
            torch.cuda.reset_peak_memory_stats()
        try:
            measurement['cpu_percent'] = self.process.cpu_percent()
            io_counters = self.process.io_counters()
            measurement['disk_read_gb'] = io_counters.read_bytes / 1024**3
            measurement['disk_write_gb'] = io_counters.write_bytes / 1024**3
        except:
            pass
        self.measurements.append(measurement)
        return measurement

    def log_measurement(self, stage: str, extra_info: str = ""):
        m = self.measure(stage)
        if m:
            gpu_str = (
                f" | GPU peak: {m['gpu_memory_peak_gb']:.2f}GB "
                f"(reserved peak: {m['gpu_memory_reserved_peak_gb']:.2f}GB)"
                if 'gpu_memory_peak_gb' in m else ""
            )
            log(
                f"  📊 [{stage}] Time: {m['time_elapsed']:.1f}s | "
                f"RAM: {m['memory_gb']:.2f}GB (Δ{m['memory_delta_gb']:+.2f})"
                f"{gpu_str}" + (f" | {extra_info}" if extra_info else "")
            )
        return m

    def save_report(self, filename: str = "resource_report.json"):
        if not self.enabled or not self.measurements:
            return
        filepath = os.path.join(self.log_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.measurements, f, indent=2, default=str)
        log(f"📊 Resource report saved to: {filepath}")

    def print_summary(self):
        if not self.enabled or not self.measurements:
            return
        total_time = self.measurements[-1]['time_elapsed']
        peak_memory = max(m['memory_gb'] for m in self.measurements)
        log(f"\n📊 RESOURCE USAGE SUMMARY:")
        log(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        log(f"   Peak RAM: {peak_memory:.2f} GB")


# ============================================================================
# ARGPARSE
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Graph Foundation Model for Particle Physics Edge Classification')
    parser.add_argument('--model', '-m', type=str, default='gcn', choices=['gcn', 'gat', 'transformer', 'sage', 'all'])
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--layers', '-l', type=int, default=6)
    parser.add_argument('--multi-scale', action='store_true',
                        help='Concatenate 1..L-hop GAT outputs and project back, '
                             'instead of residual-stacking. Requires '
                             'layers >= 1. Uses one extra Linear(hidden*L, hidden).')
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
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--mask-type', type=str, default='random', choices=['random', 'feature', 'geometry', 'cluster'])
    parser.add_argument('--mask-ratio', type=float, default=0.15)
    parser.add_argument('--mask-features', type=str, nargs='+', default=None)
    parser.add_argument('--geometry-radius', type=int, default=2)
    parser.add_argument('--pretrain-epochs', type=int, default=100)
    parser.add_argument('--finetune-epochs', type=int, default=30)
    parser.add_argument('--continuous-loss', type=str, default='mse', choices=['mse', 'l1'])

    # ---- Objective / embedding mode ----
    parser.add_argument('--objective', type=str, default='edge_classification',
                        choices=['edge_classification', 'embedding', 'cluster_slots'],
                        help='edge_classification = per-edge 5-class; '
                             'embedding = supervised contrastive + clusterer; '
                             'cluster_slots = end-to-end cluster assignment via Slot Attention')
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--min-cluster-size', type=int, default=3)
    parser.add_argument('--val-every', type=int, default=5)
    parser.add_argument('--val-events', type=int, default=15)
    parser.add_argument('--anchor-chunk-size', type=int, default=2000)
    parser.add_argument('--chunk-size', type=int, default=200,
                    help='Events preloaded per HDF5 chunk. Lower = less '
                         'peak CPU/RAM, slower per-chunk throughput.')
    parser.add_argument('--candidate-snr-column', type=int, default=2)
    parser.add_argument('--hdbscan-n-jobs', type=int, default=-1)

    # ---- Cluster-building method ----
    parser.add_argument('--cluster-method', type=str, default='hdbscan',
                        choices=['hdbscan', 'cosine_threshold', 'edge_head'])
    parser.add_argument('--cosine-threshold', type=float, default=0.8)
    parser.add_argument('--split-strict-factor', type=float, default=1.1)
    parser.add_argument('--split-min-subcluster-size', type=int, default=5)
    parser.add_argument('--split-min-cluster-size', type=int, default=2)
    parser.add_argument('--no-hierarchical-split', action='store_true')

    # ---- Two-stage edge-head ----
    parser.add_argument('--edge-head-epochs', type=int, default=10)
    parser.add_argument('--edge-head-lr', type=float, default=1e-3)
    parser.add_argument('--freeze-encoder-for-edge-head',
                        action='store_true', default=True)
    parser.add_argument('--no-freeze-encoder-for-edge-head',
                        action='store_false', dest='freeze_encoder_for_edge_head')
    parser.add_argument('--edge-head-score-threshold', type=float, default=0.5)

    # ---- Cluster-slot transformer ----
    parser.add_argument('--num-slots', type=int, default=64,
                        help='K_max: fixed slot count; empty slots collapse')
    parser.add_argument('--slot-iterations', type=int, default=3)
    parser.add_argument('--slot-temperature', type=float, default=1.0)
    parser.add_argument('--slot-hungarian', action='store_true', default=True)
    parser.add_argument('--no-slot-hungarian', action='store_false',
                        dest='slot_hungarian')
    parser.add_argument('--slot-no-object-weight', type=float, default=0.1)
    parser.add_argument('--slot-recon-weight', type=float, default=0.0)
    parser.add_argument('--slot-max-cells', type=int, default=10000,
                        help='Subsample at most this many cells per event for '
                             'the slot head. Cells not sampled are assigned to '
                             'slots by similarity in a cheap second pass. '
                             'Set to 0 to disable subsampling.')
    parser.add_argument('--slot-candidate-snr-column', type=int, default=0,
                        help='Feature column used to prioritize cells for the '
                             'slot head (default: col 2 = snr_gt2). '
                             'Set to -1 to sample uniformly at random.')
    parser.add_argument('--slot-snr-threshold', type=float, default=None,
                        help='If set, run slot attention only on cells whose '
                             'priority column >= this value. Non-surviving '
                             'cells are forced to slot 0. Takes precedence '
                             'over --slot-max-cells when both are set.')
    parser.add_argument('--limit-events', type=int, default=0,
                        help='If > 0, use only this many events (train+test). '
                             'Useful for quick overfit checks.')
    args = parser.parse_args()
    if args.multi_scale and args.model not in ('gat', 'transformer', 'all'):
        parser.error("--multi-scale requires --model gat or --model transformer")
    return args


# ============================================================================
# UTILITIES
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
            total_events = sum(
                np.lib.format.read_array_header_1_0(
                    np.lib.format.read_magic(open(lf, 'rb'))
                )[0][0] for lf in label_files
            )

    if total_events:
        results['scalability']['total_events'] = total_events
        for scale in [10_000, 50_000, 100_000, 500_000, 1_000_000]:
            total_gb_scaled = sum(
                info['total_size_gb'] / total_events * scale
                for info in results['files'].values() if info['count'] > 0
            )
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
# MASKING FOR PRETRAINING
# ============================================================================

class CalorimeterMasking:
    """
    Masking strategies for calorimeter cell graphs:
    random, feature, geometry, cluster.
    """

    def __init__(self, mask_ratio=0.15, mask_type='random',
                 mask_features=None, cells_array=None,
                 cluster_info_dict=None, geometry_radius=2):
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.mask_features = mask_features or []
        self.cells_array = cells_array
        self.cluster_info_dict = cluster_info_dict or {}
        self.geometry_radius = geometry_radius
        self.mask_token_value = 0.0

    def _get_random_mask(self, num_cells: int, rng) -> np.ndarray:
        return rng.random(num_cells) < self.mask_ratio

    def _get_feature_mask(self, num_cells: int, num_features: int,
                          feature_names: List[str], rng) -> np.ndarray:
        mask = np.zeros((num_cells, num_features), dtype=bool)
        if not self.mask_features:
            n_features_to_mask = max(1, int(num_features * self.mask_ratio))
            features_to_mask = rng.choice(num_features, n_features_to_mask, replace=False)
        else:
            features_to_mask = []
            for fname in self.mask_features:
                if fname in feature_names:
                    features_to_mask.append(feature_names.index(fname))
            if not features_to_mask:
                features_to_mask = [rng.randint(0, num_features)]
        cells_to_mask = rng.random(num_cells) < self.mask_ratio
        for feat_idx in features_to_mask:
            mask[cells_to_mask, feat_idx] = True
        return mask

    def _get_geometry_mask(self, num_cells: int, cell_positions: np.ndarray, rng) -> np.ndarray:
        seed_cell = rng.randint(0, num_cells)
        seed_eta = cell_positions[seed_cell, 0]
        seed_phi = cell_positions[seed_cell, 1]
        eta_window = 0.3 * self.geometry_radius
        phi_window = 0.3 * self.geometry_radius
        deta = np.abs(cell_positions[:, 0] - seed_eta)
        dphi = np.abs(cell_positions[:, 1] - seed_phi)
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        in_window = (deta < eta_window) & (dphi < phi_window)
        random_draw = rng.random(num_cells) < (self.mask_ratio * 2)
        return in_window & random_draw

    def _get_cluster_mask(self, num_cells: int,
                          cluster_info: Optional[dict], rng) -> np.ndarray:
        if not hasattr(self, '_cluster_mask_call_count'):
            self._cluster_mask_call_count = 0
            self._cluster_mask_fallback_count = 0
        self._cluster_mask_call_count += 1

        if cluster_info is None or 'cell_cluster_index' not in cluster_info:
            self._cluster_mask_fallback_count += 1
            if self._cluster_mask_call_count <= 5:
                log(f"  ⚠️ Cluster mask fallback "
                    f"(cluster_info={'None' if cluster_info is None else 'missing key'})")
            if self._cluster_mask_call_count % 200 == 0:
                log(f"  📊 Cluster-mask fallback rate so far: "
                    f"{self._cluster_mask_fallback_count}/"
                    f"{self._cluster_mask_call_count}")
            return self._get_random_mask(num_cells, rng)

        cluster_ids = cluster_info['cell_cluster_index']
        unique_clusters = np.unique(cluster_ids[cluster_ids > 0])
        if len(unique_clusters) == 0:
            return self._get_random_mask(num_cells, rng)
        n_clusters_to_mask = max(1, int(len(unique_clusters) * self.mask_ratio))
        clusters_to_mask = rng.choice(unique_clusters, n_clusters_to_mask, replace=False)
        return np.isin(cluster_ids, clusters_to_mask)

    def apply_mask(self, features: torch.Tensor, event_id: int = None,
                   feature_names: List[str] = None,
                   cluster_info: Optional[dict] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.RandomState()
        num_cells, num_features = features.shape

        if self.mask_type == 'random':
            cell_mask = self._get_random_mask(num_cells, rng)
            mask = np.tile(cell_mask[:, np.newaxis], (1, num_features))
        elif self.mask_type == 'feature':
            mask = self._get_feature_mask(num_cells, num_features, feature_names or [], rng)
        elif self.mask_type == 'geometry':
            if self.cells_array is not None:
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
            cell_mask = self._get_cluster_mask(num_cells, cluster_info, rng)
            mask = np.tile(cell_mask[:, np.newaxis], (1, num_features))
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

        mask = torch.from_numpy(mask).bool()
        targets = features.clone()
        masked_features = features.clone()
        masked_features[mask] = self.mask_token_value
        return masked_features, mask, targets

    def get_reconstruction_mask(self, mask: torch.Tensor, targets: torch.Tensor):
        return targets[mask], mask


# ============================================================================
# FEATURE LOADING (LAZY, HDF5-BACKED)
# ============================================================================

def load_features_lazy(data_dir: str, args, tracker: Optional[ResourceTracker] = None) -> Tuple:
    """
    Lazy feature loading: returns HDF5 references instead of pre-loading all
    events. Cluster indices are read lazily per-chunk where needed.
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

    log(f"\n📂 FILES: {len(cells_files)} cells, {len(pairs_files)} pairs, "
        f"{len(event_files)} events, {len(label_files)} labels")
    total_size_gb = sum(
        os.path.getsize(f)/1024**3
        for f in cells_files+pairs_files+event_files+label_files
        if os.path.exists(f)
    )
    log(f"   Total: {total_size_gb:.2f} GB")

    cells = np.load(cells_files[0])
    num_cells = cells.shape[0]
    log(f"  ✓ Cells: {num_cells} ({cells.nbytes/1024**2:.1f} MB)")

    pairs = np.load(pairs_files[0]).astype(np.int32)
    num_edges = pairs.shape[0]
    log(f"  ✓ Pairs: {num_edges} ({pairs.nbytes/1024**2:.1f} MB)")
    if tracker: tracker.log_measurement("static_files_loaded")

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

    feature_refs = []
    for file_idx, event_file in enumerate(event_files):
        with h5py.File(event_file, 'r') as h5f:
            if 'cell/snr_computed' in h5f:
                file_num_events = h5f['cell/snr_computed'].shape[0]
            elif 'cell/energy_raw' in h5f:
                file_num_events = h5f['cell/energy_raw'].shape[0]
            else:
                total_label_events = sum(arr.shape[0] for arr in labels)
                file_num_events = (total_label_events // len(event_files)
                                   if len(event_files) > 1 else total_label_events)

            has_eta = 'cell/cell_eta' in h5f
            has_snr = ('cell/snr_computed' in h5f
                       or 'cell/snr_raw' in h5f
                       or 'cell/energy_raw' in h5f)
            has_cluster = 'cell/cell_cluster_index' in h5f

            for local_idx in range(file_num_events):
                feature_refs.append({
                    'hdf5_path': event_file,
                    'local_idx': local_idx,
                    'has_eta': has_eta,
                    'has_snr': has_snr,
                    'has_cluster': has_cluster,
                })

    # Cluster indices needed only for embedding objective (SupCon/edge-head)
    # or cluster-mask pretraining. Supervised edge-classification skips them.
    need_cluster_info = (
        args.objective in ('embedding', 'cluster_slots')
        or (getattr(args, 'pretrain', False) and args.mask_type == 'cluster')
    )

    if not need_cluster_info:
        log("\n📊 Skipping cluster-index load (not needed for this run mode)")
        cluster_info_dict = None
    else:
        n_with_cluster = sum(1 for r in feature_refs if r.get('has_cluster', False))
        log(f"\n📊 Cluster-index loading: LAZY "
            f"({n_with_cluster}/{len(feature_refs)} events have cell_cluster_index)")
        if n_with_cluster == 0:
            log(f"  ⚠️ No events carry cell_cluster_index — embedding "
                f"objective and cluster masking will be no-ops")
        cluster_info_dict = None

    if tracker:
        tracker.log_measurement("feature_refs_built",
                                f"{len(feature_refs)} events, {input_dim} features")
        tracker.log_measurement("cluster_info_loaded",
                                f"lazy, {'needed' if need_cluster_info else 'skipped'}")

    log(f"\n✅ Lazy references built: {len(feature_refs)} events, "
        f"{num_cells} cells, {num_edges} edges, {input_dim} features")
    log(f"   RAM usage: ~{cells.nbytes/1024**2:.0f} MB (cells) + "
        f"~{pairs.nbytes/1024**2:.0f} MB (pairs) + "
        f"~{labels.nbytes/1024**2:.0f} MB (labels)")

    return (feature_refs, pairs, labels, cluster_info_dict,
            input_dim, feature_names, cells)


# ============================================================================
# LOSSES
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean', chunk_size=100000):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.reduction = reduction
        self.chunk_size = chunk_size
        self.alpha = float(alpha) if isinstance(alpha, (float, int)) else None
        self.per_class_alpha = torch.tensor(alpha, dtype=torch.float32) if isinstance(alpha, list) else None

    def forward(self, inputs, targets):
        total_loss = 0.0
        for start in range(0, inputs.size(0), self.chunk_size):
            end = min(start + self.chunk_size, inputs.size(0))
            ce = F.cross_entropy(inputs[start:end], targets[start:end],
                                 weight=self.weight, reduction='none')
            pt = torch.exp(-ce)
            alpha_t = (self.per_class_alpha.to(inputs.device)[targets[start:end]]
                       if self.per_class_alpha is not None else self.alpha)
            total_loss += (alpha_t * (1-pt)**self.gamma * ce).sum()
        return total_loss/inputs.size(0) if self.reduction == 'mean' else total_loss


def compute_class_weights(labels, num_classes, strategy='inverse', device=None, **kwargs):
    if device and device.type == 'cuda':
        counts = torch.bincount(torch.as_tensor(labels, device=device).flatten(),
                                minlength=num_classes).float()
    else:
        counts = torch.tensor(np.bincount(labels.flatten(), minlength=num_classes),
                              dtype=torch.float32)
    log(f"Class counts: {dict(zip(range(num_classes), counts.int().tolist()))}")

    if strategy == 'focal':
        total = len(labels.flatten())
        weights = kwargs.get('alpha', 0.25) * ((total-counts)/(counts+1e-5))**kwargs.get('gamma', 2.0)
    elif strategy == 'logarithmic':
        weights = 1.0/torch.log1p(counts+1e-5)
    elif strategy == 'manual':
        weights = torch.tensor([0.1, 10.0, 8.0, 8.0, 15.0], device=counts.device)[:num_classes]
    else:
        weights = 1.0/(counts+1e-5)

    weights = weights/weights.sum()*num_classes
    log(f"Computed weights ({strategy}): {weights.tolist()}")
    return weights.to(device) if device else weights


def create_loss_function(args, labels, device):
    if args.weighted_loss:
        class_weights = compute_class_weights(labels, 5, strategy=args.weight_strategy,
                                              device=device, alpha=args.focal_alpha,
                                              gamma=args.focal_gamma)
        if args.weight_strategy == 'focal':
            log(f"✅ Using Focal Loss")
            return FocalLoss(alpha=[0.10, 0.60, 0.70, 0.70, 1.00], gamma=args.focal_gamma)
        log(f"✅ Using Weighted CrossEntropyLoss")
        return nn.CrossEntropyLoss(weight=class_weights)
    log("✅ Using standard CrossEntropyLoss")
    return nn.CrossEntropyLoss()


# ---- Supervised contrastive loss (embedding objective) ----

def supervised_contrastive_loss(embeddings: torch.Tensor,
                                cluster_labels: torch.Tensor,
                                temperature: float = 0.1,
                                anchor_chunk_size: int = 2000) -> torch.Tensor:
    """
    Khosla et al. SupCon, restricted to cells with cluster_labels > 0.

    Anchors are processed in chunks against the full comparison set to keep
    peak memory at O(chunk_size * n) rather than O(n^2). Self-similarity
    exclusion uses masked_fill (out-of-place) rather than in-place indexing,
    which would invalidate the exp() backward-pass buffer.
    """
    device = embeddings.device
    valid = cluster_labels > 0
    z = embeddings[valid]
    y = cluster_labels[valid]
    n = z.shape[0]

    if n < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    z = F.normalize(z.float(), dim=-1)
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_valid_anchors = 0

    for start in range(0, n, anchor_chunk_size):
        end = min(start + anchor_chunk_size, n)
        chunk_size = end - start

        z_anchor = z[start:end]
        sim = (z_anchor @ z.T) / temperature

        local_rows = torch.arange(chunk_size, device=device)
        global_cols = torch.arange(start, end, device=device)
        self_mask = torch.zeros(chunk_size, n, dtype=torch.bool, device=device)
        self_mask[local_rows, global_cols] = True

        same = (y[start:end].unsqueeze(1) == y.unsqueeze(0))
        same = same.masked_fill(self_mask, False).float()

        logits = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits.masked_fill(self_mask, 0.0)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        n_pos = same.sum(dim=1)
        has_positive = n_pos > 0
        if not has_positive.any():
            continue
        n_pos_safe = n_pos.clamp(min=1)
        loss_per_anchor = -(same * log_prob).sum(dim=1) / n_pos_safe
        total_loss = total_loss + loss_per_anchor[has_positive].sum()
        total_valid_anchors += int(has_positive.sum().item())

    if total_valid_anchors == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / total_valid_anchors


# ---- Hungarian slot-matching loss (cluster_slots objective) ----

def hungarian_slot_loss(attn, target_clusters, no_object_weight=0.1):
    """
    DETR-style bipartite matching between K predicted slots and the unique
    ground-truth cluster IDs for one event.

    Vectorized version. Returns (loss, n_matched, matched_pairs).
    Skips the event (returns zero loss) if attn contains NaN or inf.
    """
    from scipy.optimize import linear_sum_assignment

    device = attn.device
    K, N = attn.shape
    zero = attn.sum() * 0.0

    valid = target_clusters > 0
    if not valid.any() or not torch.isfinite(attn).all():
        return zero, 0, []

    tids, inv = torch.unique(target_clusters[valid], return_inverse=True)
    T = tids.numel()
    onehot = F.one_hot(inv, T).float()                # (V, T)
    counts = onehot.sum(0).clamp(min=1)               # (T,)
    a_v = attn[:, valid]                              # (K, V)

    cost = -(a_v @ onehot) / counts                   # (K, T)
    row, col = linear_sum_assignment(cost.detach().cpu().numpy())

    slot_of = torch.full((T,), -1, device=device, dtype=torch.long)
    slot_of[torch.as_tensor(col, device=device)] = torch.as_tensor(row, device=device)
    cell_slot = slot_of[inv]
    ok = cell_slot >= 0
    cols = torch.arange(a_v.shape[1], device=device)[ok]
    logp = torch.log(a_v[cell_slot[ok], cols] + 1e-8)
    per_cluster = (-logp @ onehot[ok]) / counts
    match_loss = per_cluster.sum() / max(1, len(row))

    unmatched = torch.ones(K, dtype=torch.bool, device=device)
    unmatched[torch.as_tensor(row, device=device)] = False
    if unmatched.any():
        no_obj = attn[unmatched].sum(1).mean() / N
    else:
        no_obj = zero

    total = match_loss + no_object_weight * no_obj
    return total, int(T), list(zip(row.tolist(), col.tolist()))


# ============================================================================
# MODEL
# ============================================================================

class GraphFoundationModel(nn.Module):
    """
    Graph neural network with three task heads:
      - pretraining    : masked feature reconstruction
      - embedding      : projection head + optional edge_score_head
      - edge_classifi  : per-edge fc head (default)
      - cluster_slots  : Slot Attention head producing (K, N) assignments
    """

    def __init__(self, input_dim, hidden_dim, output_dim, device,
                 model_type='gcn', num_layers=6, num_heads=2,
                 dropout=0.0, layer_weights=False,
                 softmax_weights=False, norm_type='batch', debug=False,
                 pretraining=False, feature_names=None,
                 objective='edge_classification', embed_dim=32,
                 num_slots=64, slot_iterations=3, slot_temperature=1.0,
                 multi_scale=False,
                 slot_max_cells=0,
                 slot_candidate_snr_column=0,
                 slot_snr_threshold=None):
        super().__init__()

        self.device = device
        self.model_type = model_type
        self.num_layers = num_layers
        self.softmax = softmax_weights
        self.pretraining = pretraining
        self.feature_names = feature_names or []
        self.objective = objective
        self.embed_dim = embed_dim
        self.slot_max_cells = slot_max_cells
        self.slot_candidate_snr_column = slot_candidate_snr_column
        self.slot_snr_threshold = slot_snr_threshold

        self.node_embedding = nn.Linear(input_dim, hidden_dim)

        # ---- multi-scale guard ----
        # Concatenating per-layer outputs and projecting back only makes
        # sense for attention layers (GAT, TransformerConv), where each
        # layer learns edge-conditioned attention weights. For GCN/SAGE
        # the same code would run, but the concatenation would mean
        # something different and untested. Fail fast rather than
        # silently changing the inductive bias.
        if multi_scale and model_type not in ('gat', 'transformer'):
            raise ValueError(
                f"--multi-scale is only supported for model_type in "
                f"('gat', 'transformer'); got model_type='{model_type}'. "
                f"Drop --multi-scale or switch --model."
            )

        # ---- layer-weights / multi-scale interaction ----
        # When multi-scale is on, multi_scale_proj already learns a
        # per-scale linear mixing. Adding scalar layer-weights on top
        # double-gates the scales: the projection sees a shrunken
        # contribution, and there's no way for it to recover the
        # original magnitude. Disable layer weights in this case.
        if multi_scale and layer_weights:
            log("⚠️ --multi-scale and --layer-weights are both set. "
                "Disabling layer weights for this run — multi_scale_proj "
                "already learns per-scale mixing. Drop --layer-weights "
                "to silence this warning.")
            layer_weights = False

        self.multi_scale = multi_scale
        self.multi_scale_proj = (
            nn.Linear(hidden_dim * num_layers, hidden_dim)
            if multi_scale else None
        )
        self.layer_weights_enabled = layer_weights
        self.softmax = softmax_weights

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if model_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif model_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads,
                                          heads=num_heads, dropout=dropout, edge_dim=5))
            elif model_type == 'transformer':
                self.convs.append(TransformerConv(hidden_dim, hidden_dim // num_heads,
                                                  heads=num_heads, dropout=dropout, edge_dim=5))
            elif model_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        if norm_type == 'batch':
            self.bns = nn.ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        elif norm_type == 'layer':
            self.bns = nn.ModuleList([LayerNorm1d(hidden_dim) for _ in range(num_layers)])
        else:
            self.bns = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

        self.reconstruction_head = None
        self.projection_head = None
        self.edge_score_head = None
        self.slot_head = None
        self.fc = None

        if pretraining:
            self.reconstruction_head = FeatureReconstructionHead(
                hidden_dim, input_dim, feature_types=self._infer_feature_types())
        elif objective == 'embedding':
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            # edge_score_head consumes [emb_src, emb_dst, edge_attr] (5-wide).
            edge_in_dim = 2 * embed_dim + 5
            self.edge_score_head = nn.Sequential(
                nn.Linear(edge_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        elif objective == 'cluster_slots':
            self.slot_head = SlotAttentionClusteringHead(
                cell_dim=hidden_dim,
                slot_dim=hidden_dim,
                num_slots=num_slots,
                num_iterations=slot_iterations,
                temperature=slot_temperature,
                hidden_dim=hidden_dim,
            )
            # Optional unsupervised reconstruction term (slot-recon-weight > 0)
            self.reconstruction_head = FeatureReconstructionHead(
                hidden_dim, input_dim, feature_types=self._infer_feature_types())
        else:
            self.fc = nn.Linear(2 * hidden_dim, output_dim)

        if layer_weights:
            self.layer_weights = nn.Parameter(torch.ones(num_layers))

        mode_tag = (' [PRETRAINING]' if pretraining
                    else ' [EMBEDDING]' if objective == 'embedding'
                    else ' [CLUSTER-SLOTS]' if objective == 'cluster_slots'
                    else ' [EDGE-CLASSIFICATION]')
        log(f"📐 Initialized {model_type.upper()} model: "
            f"input={input_dim}, hidden={hidden_dim}, layers={num_layers}{mode_tag}")

    def _infer_feature_types(self) -> List[str]:
        categorical_features = {'subcalo', 'sampling', 'noise_category'}
        feature_types = []
        for fname in self.feature_names:
            if (fname in categorical_features
                    or fname.startswith('subcalo_')
                    or fname.startswith('sampling_')):
                feature_types.append('categorical')
            else:
                feature_types.append('continuous')
        return feature_types if feature_types else ['continuous']

    def encode(self, x_list, edge_index_list, edge_attr_list=None):
        """
        Batched encoder. Graphs in the list are concatenated along the node
        dimension and edge indices offset accordingly; this gives one big
        block-diagonal graph the GPU handles in a single set of kernel calls.

        Assumes every graph shares node count and edge topology (true here:
        same detector geometry every event).
        """
        weights = (
            torch.softmax(self.layer_weights, dim=0)
            if self.softmax else self.layer_weights
        ) if self.layer_weights_enabled else None

        num_graphs = len(x_list)
        num_nodes = x_list[0].shape[0]

        x_batch = torch.cat(
            [x.to(self.device, non_blocking=True) for x in x_list], dim=0)

        edge_index_batch = torch.cat(
            [edge_index_list[i].to(self.device, non_blocking=True) + i * num_nodes
             for i in range(num_graphs)],
            dim=1)

        edge_attr_batch = None
        if edge_attr_list is not None and edge_attr_list[0] is not None:
            edge_attr_batch = torch.cat(
                [e.to(self.device, non_blocking=True) for e in edge_attr_list], dim=0)

        x_embed = self.node_embedding(x_batch)

        if self.multi_scale:
            # Keep every intermediate output; each is a k-hop representation.
            # Project the concatenation back to hidden_dim so downstream
            # heads (fc, projection_head, slot_head, recon_head) see the
            # same shape they always did.
            scale_outputs = []
            h = x_embed
            for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
                if edge_attr_batch is not None and self.model_type in ['gat', 'transformer']:
                    h_new = torch.relu(bn(conv(h, edge_index_batch, edge_attr_batch)))
                else:
                    h_new = torch.relu(bn(conv(h, edge_index_batch)))
                if weights is not None:
                    h_new = weights[i] * h_new
                # Residual within each scale, same as before
                h = h + h_new
                scale_outputs.append(h)
            x_embed = self.multi_scale_proj(torch.cat(scale_outputs, dim=-1))
        else:
            for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
                if edge_attr_batch is not None and self.model_type in ['gat', 'transformer']:
                    h = torch.relu(bn(conv(x_embed, edge_index_batch, edge_attr_batch)))
                else:
                    h = torch.relu(bn(conv(x_embed, edge_index_batch)))
                if weights is not None:
                    h = weights[i] * h
                x_embed = x_embed + h

        return list(torch.split(x_embed, num_nodes, dim=0))

    def score_edges(self, node_embeddings: torch.Tensor,
                    edge_index: torch.Tensor,
                    edge_attr: torch.Tensor) -> torch.Tensor:
        """P(same-cluster) logit per edge. Requires objective='embedding'."""
        if self.edge_score_head is None:
            raise RuntimeError(
                "score_edges called but model was not built with "
                "objective='embedding'.")
        src, dst = edge_index[0], edge_index[1]
        feats = torch.cat(
            [node_embeddings[src], node_embeddings[dst], edge_attr], dim=-1)
        return self.edge_score_head(feats).squeeze(-1)

    def forward(self, x_list, edge_index_list, edge_index_out_list,
                y_batch=None, mask=None, edge_attr_list=None):
        node_embeddings = self.encode(x_list, edge_index_list, edge_attr_list)

        if self.pretraining:
            reconstructed = [self.reconstruction_head(emb) for emb in node_embeddings]
            return torch.cat(reconstructed, dim=0)

        if self.objective == 'embedding':
            return [F.normalize(self.projection_head(emb), dim=-1)
                    for emb in node_embeddings]

        if self.objective == 'cluster_slots':
            out = []
            for emb, x_raw in zip(node_embeddings, x_list):
                priorities = None
                col = self.slot_candidate_snr_column
                if col is not None and col >= 0 and col < x_raw.shape[1]:
                    priorities = x_raw[:, col].to(emb.device)
                # Run the slot head in fp32: the cell-dimension softmax
                # underflows in fp16 when N is large (~5e-6 entries), which
                # produces NaN in the einsum aggregation and crashes the
                # Hungarian solver.
                with torch.autocast(device_type='cuda', enabled=False):
                    attn, _slots = self.slot_head.forward_with_subsample(
                        emb.float(),
                        max_cells=self.slot_max_cells,
                        snr_priorities=priorities,
                        snr_threshold=self.slot_snr_threshold,
                    )
                out.append(attn.float())
            return out

        all_edge_reprs = []
        for x_embed, orig_edges in zip(node_embeddings, edge_index_out_list):
            src, dst = orig_edges[0], orig_edges[1]
            all_edge_reprs.append(torch.cat([x_embed[src], x_embed[dst]], dim=-1))
        return self.fc(torch.cat(all_edge_reprs, dim=0))


class FeatureReconstructionHead(nn.Module):
    """Predicts original node features from node embeddings (masked pretraining)."""

    def __init__(self, hidden_dim: int, output_dim: int,
                 feature_types: List[str] = None):
        super().__init__()
        self.feature_types = feature_types or ['continuous'] * output_dim
        self.reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self.categorical_heads = nn.ModuleDict()
        for i, ftype in enumerate(self.feature_types):
            if ftype == 'categorical':
                self.categorical_heads[str(i)] = nn.Linear(hidden_dim, 100)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        return self.reconstructor(node_embeddings)


class MaskedReconstructionLoss(nn.Module):
    """Combined continuous (MSE/L1) + categorical (CE) reconstruction loss."""

    def __init__(self, feature_types: List[str],
                 continuous_loss: str = 'mse',
                 categorical_weight: float = 1.0):
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
        masked_preds = predictions[mask]
        masked_targets = targets[mask]
        if masked_preds.numel() == 0:
            return torch.tensor(0.0, device=predictions.device)

        total_loss = 0.0
        for i, ftype in enumerate(self.feature_types):
            if ftype == 'continuous':
                pred = masked_preds[:, i:i+1] if masked_preds.dim() > 1 else masked_preds
                target = masked_targets[:, i:i+1] if masked_targets.dim() > 1 else masked_targets
                total_loss += self.continuous_loss_fn(pred, target).mean()
            elif ftype == 'categorical':
                pred = masked_preds[:, i]
                target = masked_targets[:, i].long()
                total_loss += self.categorical_weight * self.categorical_loss_fn(
                    pred.unsqueeze(0), target.unsqueeze(0)).mean()
        return total_loss


class SlotAttentionClusteringHead(nn.Module):
    """
    Slot Attention (Locatello et al. 2020) over graph-cell tokens.

    Produces, per cell, a distribution over K slots. Softmax is over the SLOT
    dimension (cells compete for slots), which is what distinguishes this
    from standard cross-attention (softmax over cells).
    """

    def __init__(self, cell_dim: int, slot_dim: int, num_slots: int,
                 num_iterations: int = 3, temperature: float = 1.0,
                 hidden_dim: int = 128):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.slot_dim = slot_dim

        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slot_logsigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))
        nn.init.normal_(self.slot_mu, std=1.0)

        self.to_q = nn.Linear(slot_dim, slot_dim)
        self.to_k = nn.Linear(cell_dim, slot_dim)
        self.to_v = nn.Linear(cell_dim, slot_dim)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.slot_norm = nn.LayerNorm(slot_dim)
        self.cell_norm = nn.LayerNorm(cell_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        
    def forward(self, cell_tokens: torch.Tensor):
        """
        Args:
            cell_tokens: (N, cell_dim) encoder embeddings for one event.

        Returns:
            attn:  (K, N) soft assignment; columns sum to 1.
            slots: (K, slot_dim) final slot representations.
        """
        K = self.num_slots
        # Deterministic learned queries (DETR-style). No noise during
        # training or eval, so no train/eval distribution shift.
        slots = self.slot_mu.expand(1, K, self.slot_dim).squeeze(0).contiguous().clone()

        cells = self.cell_norm(cell_tokens)
        k = self.to_k(cells)
        v = self.to_v(cells)

        attn = None
        for _ in range(self.num_iterations):
            slots_prev = slots

            q = self.to_q(self.slot_norm(slots))
            dots = torch.einsum('kd,nd->kn', q, k) / (self.slot_dim ** 0.5)

            # Softmax over SLOTS (dim=0): cells' mass competes across slots.
            attn = torch.softmax(dots / self.temperature, dim=0)

            # Normalize over cells so each slot gets a weighted average.
            attn_norm = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
            updates = torch.einsum('kn,nd->kd', attn_norm, v)

            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(K, self.slot_dim)
            slots = slots + self.mlp(slots)

        q = self.to_q(self.slot_norm(slots))
        dots = torch.einsum('kd,nd->kn', q, k) / (self.slot_dim ** 0.5)
        attn = torch.softmax(dots / self.temperature, dim=0)

        return attn, slots

    def _assign_all_cells_to_slots(self, cell_tokens, slots, chunk_size=8192):
        """
        Assign every cell to a slot using q.k similarity only. No gradient,
        no iteration. Memory is bounded by chunk_size.
        """
        cells = self.cell_norm(cell_tokens)
        k_full = self.to_k(cells)
        q = self.to_q(self.slot_norm(slots))
        scale = self.slot_dim ** 0.5

        N = cell_tokens.shape[0]
        attn_chunks = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            dots = torch.einsum('kd,nd->kn', q, k_full[start:end]) / scale
            attn_chunks.append(
                torch.softmax(dots / self.temperature, dim=0))
        return torch.cat(attn_chunks, dim=1)

    def forward_with_subsample(self, cell_tokens, max_cells=0,
                               snr_priorities=None, snr_threshold=None):
        """
        Run slot attention on at most `max_cells` tokens, then assign the
        remaining tokens to the nearest slot via q.k similarity.

        Args:
            cell_tokens:     (N, cell_dim)
            max_cells:       int; if <= 0 or N <= max_cells, runs the
                             standard forward on all tokens.
            snr_priorities:  optional (N,) float tensor. If provided, the
                             top-`max_cells` tokens by this score form the
                             subset. Ties are broken arbitrarily.
            snr_threshold:   optional float. If set, tokens with priority
                             >= threshold form the subset; the rest are
                             assigned by similarity without forcing to slot 0.

        Returns:
            attn:  (K, N) soft assignment over ALL tokens.
            slots: (K, slot_dim) final slot representations.
        """
        N = cell_tokens.shape[0]
        device = cell_tokens.device

        # --- Threshold mode: pick subset by priority >= threshold ---
        if snr_threshold is not None and snr_priorities is not None:
            prio = snr_priorities.to(device).float().flatten()
            if prio.numel() == N:
                keep = prio >= snr_threshold
                if keep.sum() == 0:
                    keep = torch.zeros_like(keep)
                    keep[torch.argmax(prio)] = True
                sub_tokens = cell_tokens[keep]
                _, slots = self.forward(sub_tokens)
                attn = self._assign_all_cells_to_slots(cell_tokens, slots)
                return attn, slots

        # --- Subsample mode: keep only the top-M cells for slot attention ---
        if max_cells is not None and max_cells > 0 and N > max_cells:
            if snr_priorities is not None:
                priorities = snr_priorities.to(device).float().flatten()
                if priorities.numel() == N:
                    _, idx = torch.topk(priorities, k=max_cells, largest=True)
                else:
                    idx = torch.randperm(N, device=device)[:max_cells]
            else:
                idx = torch.randperm(N, device=device)[:max_cells]
            idx = idx.sort().values

            sub_tokens = cell_tokens[idx]                 # (M, cell_dim)
            _, slots = self.forward(sub_tokens)           # slot head on subset

            attn = self._assign_all_cells_to_slots(cell_tokens, slots)
            return attn, slots

        # --- No subsampling: use all cells ---
        return self.forward(cell_tokens)

        device = cell_tokens.device

# ============================================================================
# DATA
# ============================================================================

class MultiClassBatchGenerator(IterableDataset):
    """
    Streams graph samples in chunks. Features (and cluster indices when
    present) are read on demand from HDF5; cluster_info_dict is now a
    deprecated no-op kept for call-site compatibility.
    """

    def __init__(self, feature_refs, neighbor_pairs, labels,
                 cells_array, mode="train", is_bi_directional=True,
                 batch_size=1, train_ratio=0.7, debug=False,
                 cluster_info_dict=None, chunk_size=2000, inference_only=False,
                 limit_events=0):
        self.debug = debug
        self.batch_size = batch_size
        self.mode = mode
        self.num_pairs = neighbor_pairs.shape[0]
        self.chunk_size = chunk_size
        self.inference_only = inference_only
        self.is_bi_directional = is_bi_directional
        self._cells_array = cells_array
        self._feature_refs = feature_refs
        self._labels = labels

        self.neighbor_pairs = torch.as_tensor(neighbor_pairs, dtype=torch.long)
        raw_edges = self.neighbor_pairs.T.contiguous()

        if self.is_bi_directional:
            self.pairs_mp = to_undirected(raw_edges)
            self.pairs_pred = raw_edges
        else:
            self.pairs_mp = raw_edges
            self.pairs_pred = raw_edges

        # edge_attr (built as 2 * pairs_pred.shape[1] rows) must line up
        # with pairs_mp: to_undirected() can deduplicate and silently break
        # GAT/Transformer's edge_attr alignment. Fail fast here.
        if self.is_bi_directional:
            expected_mp_edges = 2 * self.pairs_pred.shape[1]
            actual_mp_edges = self.pairs_mp.shape[1]
            if actual_mp_edges != expected_mp_edges:
                raise ValueError(
                    f"Edge count mismatch: pairs_mp has {actual_mp_edges} edges "
                    f"but edge_attr will be built with {expected_mp_edges} rows "
                    f"(2 * {self.pairs_pred.shape[1]} pairs_pred edges). "
                    f"to_undirected() likely deduplicated repeated pairs."
                )

        if torch.cuda.is_available():
            self.pairs_mp = self.pairs_mp.pin_memory()
            self.pairs_pred = self.pairs_pred.pin_memory()

        self.num_events = len(feature_refs)
        all_event_ids = list(range(self.num_events))
        split_idx = int(self.num_events * train_ratio)

        if mode == "train":
            self.event_indices = all_event_ids[:split_idx]
        else:
            self.event_indices = all_event_ids[split_idx:]

        if limit_events and limit_events > 0:
            self.event_indices = self.event_indices[:limit_events]

        pin_msg = " [pinned]" if torch.cuda.is_available() else ""
        log(f"📊 {mode.upper()} SET: {len(self.event_indices)} events "
            f"[CUDA, chunk={chunk_size}, bidirectional={is_bi_directional}{pin_msg}]")

        self.num_chunks = (len(self.event_indices) + chunk_size - 1) // chunk_size
        self._chunk_data = []

    def _compute_edge_features_from_array(self, features_np):
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
        local_idx = ref['local_idx']

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

        if ref['has_eta']:
            eta_row = h5f['cell/cell_eta'][local_idx]
            phi_row = h5f['cell/cell_phi'][local_idx]
        else:
            eta_row = self._cells_array['eta_event0'].astype(np.float32)
            phi_row = self._cells_array['phi_event0'].astype(np.float32)

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
        if self.debug or num_chunks <= 1 or chunk_idx % max(1, num_chunks // 5) == 0:
            log(f"  📂 Chunk {chunk_idx+1}/{num_chunks}: "
                f"events {chunk_events[0]}-{chunk_events[-1]} ({len(chunk_events)})")

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

            # Lazy cluster-index read: reuse the already-open handle.
            cluster_info = None
            if ref.get('has_cluster', False):
                cidx = current_h5f['cell/cell_cluster_index'][ref['local_idx']]
                cluster_info = {'cell_cluster_index': cidx}

            chunk_samples.append((
                x_scaled, self.pairs_mp, self.pairs_pred,
                out_labels, edge_attr, cluster_info, event_idx
            ))

        if current_h5f is not None:
            current_h5f.close()

        gc.collect()
        return chunk_samples

    def _free_chunk(self):
        if self._chunk_data:
            for sample in self._chunk_data:
                del sample
            del self._chunk_data
            self._chunk_data = []
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
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
        return len(self.event_indices)

    @staticmethod
    def collate_data(batch):
        """Batch elements are 7-tuples: (x, pairs_mp, pairs_pred, labels, edge_attr, cluster_info, event_idx)."""
        return (
            [b[0] for b in batch],
            [b[1] for b in batch],
            [b[2] for b in batch],
            torch.cat([b[3] for b in batch], dim=0),
            [b[4] for b in batch] if batch[0][4] is not None else None,
            [b[5] for b in batch],
            [b[6] for b in batch],
        )


# ============================================================================
# TRAINING LOOPS
# ============================================================================

def pretrain_epoch(model, loader, optimizer, criterion, masking_fn,
                   scaler, device, feature_names, debug=False):
    """One epoch of masked reconstruction pretraining."""
    model.train()
    total_loss = 0
    total_masked = 0
    optimizer.zero_grad(set_to_none=True)
    use_amp = scaler is not None

    for batch_idx, batch in enumerate(loader):
        x_list, ei_list, eio_list, _, edge_attr_list, cluster_infos, event_ids = batch

        masked_x_list = []
        mask_list = []
        targets_list = []
        for i, x in enumerate(x_list):
            if masking_fn:
                masked_x, mask, targets = masking_fn.apply_mask(
                    x,
                    event_id=event_ids[i] if event_ids else None,
                    feature_names=feature_names,
                    cluster_info=cluster_infos[i],
                )
                masked_x_list.append(masked_x)
                mask_list.append(mask)
                targets_list.append(targets)
            else:
                masked_x_list.append(x)

        masked_x_list = [x.to(device, non_blocking=True) for x in masked_x_list]
        ei_list = [e.to(device, non_blocking=True) for e in ei_list]
        mask_list = [m.to(device, non_blocking=True) for m in mask_list]
        targets_list = [t.to(device, non_blocking=True) for t in targets_list]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            predictions = model(masked_x_list, ei_list, eio_list, edge_attr_list=edge_attr_list)

            # predictions is one concatenated tensor over all graphs in the
            # batch; split per-graph by node count so every sample
            # contributes to the loss.
            node_counts = [t.shape[0] for t in targets_list]
            pred_list = list(torch.split(predictions, node_counts, dim=0))

            loss = 0
            for pred, target, mask in zip(pred_list, targets_list, mask_list):
                loss += criterion(pred, target, mask)
            loss /= len(pred_list)

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
    """One epoch of edge classification training (FP16 + grad accumulation)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad(set_to_none=True)
    use_amp = scaler is not None

    for batch_idx, batch in enumerate(loader):
        x_list, ei_list, eio_list, y_batch, edge_attr_list, _, _ = batch

        x_list = [x.to(device, non_blocking=True) for x in x_list]
        ei_list = [e.to(device, non_blocking=True) for e in ei_list]
        eio_list = [e.to(device, non_blocking=True) for e in eio_list]
        y_batch = y_batch.to(device, non_blocking=True).squeeze(1)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            scores = model(x_list, ei_list, eio_list, edge_attr_list=edge_attr_list)
            loss = criterion(scores, y_batch) / accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.float().item() * len(y_batch) * accumulation_steps
        preds = scores.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return {
        "loss": total_loss / total if total else 0,
        "acc": correct / total if total else 0,
    }


def compute_metrics_from_numpy(y_true, y_pred, total_loss, total_samples, num_classes=5):
    """Vectorized confusion matrix + per-class / macro / weighted metrics."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    cm_sum = cm.sum()

    if cm_sum == 0:
        zero_dict = {
            "loss": 0, "accuracy": 0, "macro_f1": 0, "f1_sum_score": 0,
            "randomness_metric": 0, "macro_recall": 0, "macro_precision": 0,
            "weighted_recall": 0, "weighted_precision": 0, "weighted_f1": 0,
            "weighted_recall_score": 0, "weighted_f1_score": 0,
            "class_totals": [0] * num_classes,
            "confusion_matrix": cm.tolist(),
        }
        for c in range(num_classes):
            zero_dict[f"recall_class_{c}"] = 0.0
            zero_dict[f"precision_class_{c}"] = 0.0
            zero_dict[f"f1_class_{c}"] = 0.0
        return zero_dict

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    class_totals = cm.sum(axis=1)
    class_weights = class_totals / cm_sum

    recall = np.divide(TP, TP + FN, out=np.zeros(num_classes), where=(TP + FN) > 0)
    precision = np.divide(TP, TP + FP, out=np.zeros(num_classes), where=(TP + FP) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(num_classes), where=(precision + recall) > 0)
    accuracy = TP.sum() / cm_sum

    return {
        "loss": total_loss / total_samples if total_samples else 0,
        "accuracy": accuracy,
        **{f"recall_class_{c}": recall[c] for c in range(num_classes)},
        **{f"precision_class_{c}": precision[c] for c in range(num_classes)},
        **{f"f1_class_{c}": f1[c] for c in range(num_classes)},
        "macro_recall": np.mean(recall),
        "macro_precision": np.mean(precision),
        "macro_f1": np.mean(f1),
        "weighted_recall": np.sum(class_weights * recall),
        "weighted_precision": np.sum(class_weights * precision),
        "weighted_f1": np.sum(class_weights * f1),
        "randomness_metric": np.mean(recall) * num_classes,
        "f1_sum_score": np.mean(f1) * num_classes,
        "weighted_recall_score": np.sum(class_weights * recall) * num_classes,
        "weighted_f1_score": np.sum(class_weights * f1) * num_classes,
        "class_totals": class_totals.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def evaluate(model, loader, criterion, device, use_amp=False, num_classes=5):
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x_list, ei_list, eio_list, y_batch, edge_attr_list, _, _ in loader:
            x_list = [x.to(device, non_blocking=True) for x in x_list]
            ei_list = [e.to(device, non_blocking=True) for e in ei_list]
            eio_list = [e.to(device, non_blocking=True) for e in eio_list]
            y_batch = y_batch.to(device, non_blocking=True).squeeze(1)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                scores = model(x_list, ei_list, eio_list, edge_attr_list=edge_attr_list)

            total_loss += criterion(scores.float(), y_batch).item() * len(y_batch)
            preds = scores.argmax(dim=1)
            all_labels.append(y_batch.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            total += len(y_batch)

    y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    return compute_metrics_from_numpy(y_true, y_pred, total_loss, total, num_classes)


def train_epoch_embedding(model, loader, optimizer, scaler, device,
                          temperature=0.1, accumulation_steps=1, debug=False,
                          anchor_chunk_size=2000):
    """One epoch of supervised contrastive training on node embeddings."""
    model.train()
    total_loss = 0.0
    total_events = 0
    events_with_cluster_info = 0
    use_amp = scaler is not None
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        x_list, ei_list, _, _, edge_attr_list, cluster_infos, _ = batch

        x_list = [x.to(device, non_blocking=True) for x in x_list]
        ei_list = [e.to(device, non_blocking=True) for e in ei_list]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            emb_list = model(x_list, ei_list, [None] * len(x_list),
                             edge_attr_list=edge_attr_list)

            loss = torch.tensor(0.0, device=device)
            n_used = 0
            for emb, cinfo in zip(emb_list, cluster_infos):
                events_with_cluster_info += 1
                if cinfo is None or 'cell_cluster_index' not in cinfo:
                    continue
                cidx = torch.as_tensor(cinfo['cell_cluster_index'], device=device).long()
                l = supervised_contrastive_loss(
                    emb, cidx, temperature=temperature,
                    anchor_chunk_size=anchor_chunk_size)
                if l.requires_grad:
                    loss = loss + l
                    n_used += 1

            if n_used == 0:
                optimizer.zero_grad(set_to_none=True)
                continue
            loss = loss / n_used / accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.float().item() * accumulation_steps
        total_events += n_used

        if debug and batch_idx >= 2:
            break

    return {
        "loss": total_loss / max(1, total_events),
        "n_events": total_events,
        "n_events_seen": events_with_cluster_info,
    }


def train_epoch_edge_head(model, loader, optimizer, device,
                          accumulation_steps=1, debug=False,
                          freeze_encoder=False):
    """
    Stage 2: train edge_score_head to predict P(same cluster) per edge.

    Uses pairs_pred (un-doubled original edges) so the head sees the same
    edge set inference-time clustering will score. Any pair touching a noise
    cell (cluster index <= 0) is an explicit negative.

    edge_attr is [forward_edges..., reverse_edges...]; forward rows align
    with pairs_pred, so slice to the first len(eio) rows.
    """
    model.train()
    if freeze_encoder:
        for name, module in model.named_children():
            if name not in ('edge_score_head',):
                module.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_events = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        x_list, ei_list, eio_list, _, edge_attr_list, cluster_infos, _ = batch
        x_list = [x.to(device, non_blocking=True) for x in x_list]
        ei_list = [e.to(device, non_blocking=True) for e in ei_list]
        eio_list = [e.to(device, non_blocking=True) for e in eio_list]

        emb_list = model(x_list, ei_list, [None] * len(x_list),
                         edge_attr_list=edge_attr_list)

        loss = torch.tensor(0.0, device=device)
        n_used = 0
        for emb, eio, ea, cinfo in zip(emb_list, eio_list, edge_attr_list, cluster_infos):
            if cinfo is None or 'cell_cluster_index' not in cinfo:
                continue
            if ea is None:
                continue
            ea_dev = ea.to(device, non_blocking=True)
            ea_fwd = ea_dev[:eio.shape[1]]
            cidx = torch.as_tensor(cinfo['cell_cluster_index'], device=device).long()
            src, dst = eio[0], eio[1]
            target = ((cidx[src] == cidx[dst]) & (cidx[src] > 0)).float()
            logits = model.score_edges(emb, eio, ea_fwd)
            loss = loss + criterion(logits, target)
            n_used += 1

        if n_used == 0:
            optimizer.zero_grad(set_to_none=True)
            continue

        loss = loss / n_used / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        total_events += n_used

        if debug and batch_idx >= 2:
            break

    return {"loss": total_loss / max(1, total_events), "n_events": total_events}


def train_epoch_cluster_slots(model, loader, optimizer, scaler, device,
                              no_object_weight=0.1, accumulation_steps=1,
                              debug=False, recon_weight=0.0):
    """
    One epoch of end-to-end cluster-slot training via Hungarian matching.
    Optional reconstruction term (recon_weight > 0) adds an unsupervised
    signal that also works without ground-truth clusters.
    """
    model.train()
    total_match = 0.0
    total_recon = 0.0
    total_events = 0
    use_amp = scaler is not None
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        x_list, ei_list, _, _, edge_attr_list, cluster_infos, _ = batch
        x_list = [x.to(device, non_blocking=True) for x in x_list]
        ei_list = [e.to(device, non_blocking=True) for e in ei_list]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            attn_list = model(x_list, ei_list, [None] * len(x_list),
                              edge_attr_list=edge_attr_list)

            loss_match = torch.tensor(0.0, device=device)
            n_used = 0
            for attn, cinfo in zip(attn_list, cluster_infos):
                if cinfo is None or 'cell_cluster_index' not in cinfo:
                    continue
                # attn is fp16 under autocast; log/softmax numerics need fp32.
                attn = attn.float()
                targets = torch.as_tensor(cinfo['cell_cluster_index'], device=device).long()
                l, n_matched, _ = hungarian_slot_loss(
                    attn, targets, no_object_weight=no_object_weight)
                if n_matched == 0:
                    continue
                loss_match = loss_match + l
                n_used += 1

            if n_used == 0:
                optimizer.zero_grad(set_to_none=True)
                continue

            loss = loss_match / n_used

            if recon_weight > 0.0:
                recon_loss = torch.tensor(0.0, device=device)
                for emb, x_raw in zip(
                    model.encode(x_list, ei_list, edge_attr_list), x_list
                ):
                    pred = model.reconstruction_head(emb)
                    recon_loss = recon_loss + F.mse_loss(pred, x_raw)
                recon_loss = recon_loss / len(x_list)
                loss = loss + recon_weight * recon_loss
                total_recon += recon_loss.float().item()

            loss = loss / accumulation_steps

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_match += loss_match.float().item() * accumulation_steps
        total_events += n_used

        if debug and batch_idx >= 2:
            break

    return {
        'match_loss': total_match / max(1, total_events),
        'recon_loss': total_recon / max(1, total_events),
        'n_events': total_events,
    }


# ============================================================================
# BATCHED INFERENCE (EDGE CLASSIFICATION)
# ============================================================================

@torch.no_grad()
def run_inference(model, generator, device, criterion=None, num_classes=5,
                  debug=False, show_progress=True, save_path=None,
                  model_name=None):
    """Edge classification inference over a dataset; optionally computes metrics."""
    model.eval()
    batch_results = []
    batch_size = 100
    total_events = len(generator)
    total_loss = 0.0
    total_samples = 0
    all_labels_list = []
    all_preds_list = []

    if show_progress and not debug:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(generator), total=total_events,
                            desc="   🔮 Inference", unit="events")
        except ImportError:
            iterator = enumerate(generator)
    else:
        iterator = enumerate(generator)

    for i, (x_scaled, edge_index, edge_index_out, y,
            edge_attr, cluster_info, _event_idx) in iterator:

        x_scaled = x_scaled.to(device, non_blocking=True)
        edge_index = edge_index.to(device, non_blocking=True)
        edge_index_out = edge_index_out.to(device, non_blocking=True)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device, non_blocking=True)

        out = model([x_scaled], [edge_index], [edge_index_out],
                    edge_attr_list=[edge_attr] if edge_attr is not None else None)

        preds = out.argmax(dim=1).cpu().numpy()
        scores = torch.softmax(out, dim=1).cpu().numpy()
        src_nodes = edge_index_out[0].cpu().numpy()
        dst_nodes = edge_index_out[1].cpu().numpy()

        labels_np = (y.squeeze(1).numpy()
                     if y is not None and y.dim() == 2
                     else (y.numpy() if y is not None else None))

        if criterion is not None and labels_np is not None:
            y_tensor = y.to(device, non_blocking=True).squeeze(1)
            loss_val = criterion(out, y_tensor).float().item() * len(y_tensor)
            total_loss += loss_val
            total_samples += len(y_tensor)
            all_labels_list.append(labels_np)
            all_preds_list.append(preds)

        batch_results.append({
            "event_id": (generator.event_indices[i]
                         if hasattr(generator, 'event_indices') else i),
            "preds": preds,
            "scores": scores,
            "labels": labels_np,
            "neighbor_pairs": np.stack([src_nodes, dst_nodes], axis=1),
            "cluster_info": cluster_info,
        })

        if len(batch_results) >= batch_size and save_path:
            log(f"  💾 Saving batch ({i+1}/{total_events})...")
            save_results_to_parquet(batch_results, save_path, model_name, append=True)
            batch_results = []
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if debug and i >= 4:
            log("Debug: stopping after 5 events")
            break

    if batch_results and save_path:
        save_results_to_parquet(batch_results, save_path, model_name, append=True)

    metrics_dict = None
    if criterion is not None and all_labels_list:
        y_true = np.concatenate(all_labels_list)
        y_pred = np.concatenate(all_preds_list)
        metrics_dict = compute_metrics_from_numpy(y_true, y_pred, total_loss,
                                                  total_samples, num_classes)

    return batch_results, metrics_dict


# ============================================================================
# PARQUET SAVING
# ============================================================================

def save_results_to_parquet(results, save_path, model_name,
                            cluster_info_dict=None, append=False):
    """Flatten inference outputs into an edge-level Parquet table."""
    if not results:
        return ""

    if not append and os.path.exists(save_path):
        os.remove(save_path)
    if not append:
        log("   💾 Saving to Parquet...")

    writer = None
    if append and os.path.exists(save_path):
        existing_schema = pq.ParquetFile(save_path).schema_arrow

    has_cluster = bool(results and results[0].get('cluster_info'))

    for chunk_start in range(0, len(results), 10):
        chunk_end = min(chunk_start + 10, len(results))
        chunk_results = results[chunk_start:chunk_end]

        chunk_edges = sum(len(r['preds']) for r in chunk_results)
        event_ids = np.zeros(chunk_edges, dtype=np.int32)
        edge_ids = np.zeros(chunk_edges, dtype=np.int32)
        src = np.zeros(chunk_edges, dtype=np.int32)
        dst = np.zeros(chunk_edges, dtype=np.int32)
        true_labels = np.full(chunk_edges, -1, dtype=np.int8)
        pred_labels = np.zeros(chunk_edges, dtype=np.int8)
        confidence = np.zeros(chunk_edges, dtype=np.float32)
        scores = np.zeros((chunk_edges, 5), dtype=np.float32)

        if has_cluster:
            src_cluster = np.full(chunk_edges, -1, dtype=np.int32)
            dst_cluster = np.full(chunk_edges, -1, dtype=np.int32)

        offset = 0
        for result in chunk_results:
            n = len(result['preds'])
            event_ids[offset:offset+n] = result['event_id']
            edge_ids[offset:offset+n] = np.arange(n)
            src[offset:offset+n] = result['neighbor_pairs'][:, 0]
            dst[offset:offset+n] = result['neighbor_pairs'][:, 1]
            if result['labels'] is not None:
                true_labels[offset:offset+n] = result['labels']
            pred_labels[offset:offset+n] = result['preds']
            confidence[offset:offset+n] = result['scores'][
                np.arange(n), result['preds']]
            scores[offset:offset+n] = result['scores']

            if has_cluster:
                cidx = result['cluster_info'].get('cell_cluster_index')
                if cidx is not None:
                    src_cluster[offset:offset+n] = cidx[result['neighbor_pairs'][:, 0]]
                    dst_cluster[offset:offset+n] = cidx[result['neighbor_pairs'][:, 1]]
            offset += n

        df = pd.DataFrame({
            'event_id': event_ids, 'edge_id': edge_ids,
            'source_id': src, 'target_id': dst,
            'true_label': true_labels, 'pred_label': pred_labels,
            'confidence': confidence,
            'score_class_0': scores[:, 0], 'score_class_1': scores[:, 1],
            'score_class_2': scores[:, 2], 'score_class_3': scores[:, 3],
            'score_class_4': scores[:, 4], 'model_name': model_name,
        })

        if has_cluster:
            df['source_cluster'] = src_cluster
            df['target_cluster'] = dst_cluster
            df['same_cluster'] = (src_cluster == dst_cluster) & (src_cluster > 0)

        table = pa.Table.from_pandas(df, preserve_index=False)

        if writer is None:
            schema = (existing_schema
                      if (append and os.path.exists(save_path))
                      else table.schema)
            writer = pq.ParquetWriter(save_path, schema,
                                      compression='zstd', compression_level=3,
                                      use_dictionary=True, write_statistics=True)
        writer.write_table(table)
        del df, table
        gc.collect()

    if writer:
        writer.close()
    if not append:
        log(f"   💾 Saved: {save_path} "
            f"({os.path.getsize(save_path)/1024**3:.2f} GB)")
    return save_path


# ============================================================================
# CLUSTER-BUILDING (EMBEDDING MODE)
# ============================================================================

def build_clusters_cosine_threshold(embeddings, pairs_pred, cosine_threshold=0.8,
                                    do_split=True, split_strict_factor=1.1,
                                    split_min_subcluster_size=5,
                                    split_min_cluster_size=2, debug=False):
    """Cosine threshold + union-find (+ optional hierarchical split).
    Embeddings are L2-normalized; cosine is a plain dot product. Score is
    remapped to [0,1] via (cos+1)/2 so 0.5 = uncorrelated."""
    n_cells = embeddings.shape[0]
    src = pairs_pred[0].numpy()
    dst = pairs_pred[1].numpy()

    cos_sim = (embeddings[src] * embeddings[dst]).sum(axis=1)
    score = (cos_sim + 1.0) * 0.5

    if debug:
        log(f"    Cosine score: min={cos_sim.min():.3f} max={cos_sim.max():.3f} "
            f"mean={cos_sim.mean():.3f} | edges above thresh: "
            f"{(score >= cosine_threshold).sum()}/{len(score)}")

    edges_df = pd.DataFrame({
        'source_id': src.astype('int64'),
        'target_id': dst.astype('int64'),
        'score_class_1': score.astype('float64'),
    })
    mask = edges_df['score_class_1'].values >= cosine_threshold

    if not do_split:
        return build_clusters_from_mask(
            edges_df, mask, n_cells, min_size=split_min_cluster_size).astype(np.int64)
    return hierarchical_split(
        edges_df, mask, n_cells,
        strict_factor=split_strict_factor,
        min_subcluster_size=split_min_subcluster_size).astype(np.int64)


def build_clusters_edge_head(model, node_embeddings_np, pairs_pred,
                             edge_attr_fwd, device, score_threshold=0.5,
                             do_split=True, split_strict_factor=1.1,
                             split_min_subcluster_size=5,
                             split_min_cluster_size=2, debug=False):
    """Learned-head variant of cosine threshold. Embeds must be L2-normalized
    (the same ones the head was trained against)."""
    n_cells = node_embeddings_np.shape[0]
    src = pairs_pred[0].numpy()
    dst = pairs_pred[1].numpy()

    emb_t = torch.as_tensor(node_embeddings_np, dtype=torch.float32, device=device)
    eio_t = pairs_pred.to(device)
    ea_t = edge_attr_fwd.to(device)

    with torch.no_grad():
        logits = model.score_edges(emb_t, eio_t, ea_t)
        probs = torch.sigmoid(logits).float().cpu().numpy()

    if debug:
        log(f"    Edge-head probs: min={probs.min():.3f} max={probs.max():.3f} "
            f"mean={probs.mean():.3f} | above thresh: "
            f"{(probs >= score_threshold).sum()}/{len(probs)}")

    edges_df = pd.DataFrame({
        'source_id': src.astype('int64'),
        'target_id': dst.astype('int64'),
        'score_class_1': probs.astype('float64'),
    })
    mask = edges_df['score_class_1'].values >= score_threshold

    if not do_split:
        return build_clusters_from_mask(
            edges_df, mask, n_cells, min_size=split_min_cluster_size).astype(np.int64)
    return hierarchical_split(
        edges_df, mask, n_cells,
        strict_factor=split_strict_factor,
        min_subcluster_size=split_min_subcluster_size).astype(np.int64)


# ============================================================================
# EMBEDDING-MODE INFERENCE
# ============================================================================

@torch.no_grad()
def run_embedding_inference(model, generator, device,
                            min_cluster_size=3, debug=False,
                            show_progress=True, max_events=None,
                            candidate_snr_column=2, hdbscan_n_jobs=-1,
                            cluster_method='hdbscan', cosine_threshold=0.8,
                            split_strict_factor=1.1, split_min_subcluster_size=5,
                            split_min_cluster_size=2, no_hierarchical_split=False,
                            edge_head_score_threshold=0.5):
    """Extract per-event embeddings and cluster them via the selected method."""
    from sklearn.cluster import HDBSCAN

    if (cluster_method in ('cosine_threshold', 'edge_head')
            and not no_hierarchical_split and not SPLIT_AVAILABLE):
        raise RuntimeError(
            f"cluster_method='{cluster_method}' with hierarchical split "
            f"requires hierarchical_split_reattach.py to be importable.")

    model.eval()
    results = []
    total_events = (min(max_events, len(generator))
                    if max_events is not None else len(generator))

    iterator = enumerate(generator)
    if show_progress and not debug and max_events is None:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(generator), total=total_events,
                            desc=f"   🔮 Emb inference [{cluster_method}]",
                            unit="events")
        except ImportError:
            pass

    for i, (x, ei, eio, _, edge_attr, cinfo, event_idx) in iterator:
        x_dev = x.to(device, non_blocking=True)
        ei = ei.to(device, non_blocking=True)
        ea = (edge_attr.to(device, non_blocking=True)
              if edge_attr is not None else None)

        emb_list = model([x_dev], [ei], [None],
                         edge_attr_list=[ea] if ea is not None else None)
        emb_full = emb_list[0].float().cpu().numpy()
        n_cells = emb_full.shape[0]

        if candidate_snr_column is not None:
            candidate_mask = x[:, candidate_snr_column].numpy() > 0.5
        else:
            candidate_mask = np.ones(n_cells, dtype=bool)

        pred = np.zeros(n_cells, dtype=np.int64)
        n_candidates = int(candidate_mask.sum())

        if cluster_method == 'hdbscan':
            if n_candidates >= min_cluster_size:
                sub_emb = emb_full[candidate_mask]
                try:
                    clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                                        metric='euclidean', n_jobs=hdbscan_n_jobs)
                except TypeError:
                    clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                                        metric='euclidean')
                sub_pred = clusterer.fit_predict(sub_emb)
                sub_pred = sub_pred.astype(np.int64) + 1
                sub_pred[sub_pred <= 0] = 0
                pred[candidate_mask] = sub_pred

        elif cluster_method == 'cosine_threshold':
            # Zeroing non-candidate embeddings forces their cosine similarity
            # to every other cell to be 0 (mathematically dissimilar). This
            # trick is valid for cosine similarity, unlike for the learned
            # edge head which was trained on real embeddings.
            emb_masked = emb_full.copy()
            emb_masked[~candidate_mask] = 0.0
            pred = build_clusters_cosine_threshold(
                emb_masked, eio,
                cosine_threshold=cosine_threshold,
                do_split=not no_hierarchical_split,
                split_strict_factor=split_strict_factor,
                split_min_subcluster_size=split_min_subcluster_size,
                split_min_cluster_size=split_min_cluster_size,
                debug=debug)

        elif cluster_method == 'edge_head':
            # Do NOT zero non-candidate embeddings here. Zeroing is a
            # cosine-specific trick (zero vector has 0 dot product with
            # anything); the learned head was trained on real embeddings
            # of all cells and would be evaluated out-of-distribution.
            ea_fwd = (ea[:eio.shape[1]] if ea is not None else None)
            if ea_fwd is None:
                raise RuntimeError("edge_head cluster method requires edge_attr.")
            pred = build_clusters_edge_head(
                model, emb_full, eio, ea_fwd, device,
                score_threshold=edge_head_score_threshold,
                do_split=not no_hierarchical_split,
                split_strict_factor=split_strict_factor,
                split_min_subcluster_size=split_min_subcluster_size,
                split_min_cluster_size=split_min_cluster_size,
                debug=debug)
        else:
            raise ValueError(f"Unknown cluster_method: {cluster_method}")

        truth = None
        if cinfo is not None and 'cell_cluster_index' in cinfo:
            truth = np.asarray(cinfo['cell_cluster_index'], dtype=np.int64)

        results.append({
            "event_id": event_idx,
            "pred_labels": pred,
            "truth_labels": truth,
            "embeddings": emb_full,
            "cluster_info": cinfo,
            "n_candidates": n_candidates,
        })

        if max_events is not None and len(results) >= max_events:
            break
        if debug and i >= 4:
            log("Debug: stopping after 5 events")
            break

    return results


# ============================================================================
# CLUSTER-SLOT INFERENCE
# ============================================================================

@torch.no_grad()
def run_cluster_slot_inference(model, generator, device, debug=False,
                               show_progress=True, max_events=None):
    """
    Per-event forward pass through Slot Attention; argmax over slots gives
    per-cell cluster labels. Also returns per-slot cell counts (slot_usage)
    for collapse diagnostics.
    """
    model.eval()
    results = []
    total_events = (min(max_events, len(generator))
                    if max_events is not None else len(generator))

    iterator = enumerate(generator)
    if show_progress and not debug and max_events is None:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(generator), total=total_events,
                            desc="   🔮 Slot inference", unit="events")
        except ImportError:
            pass

    for i, (x, ei, _eio, _y, edge_attr, cinfo, event_idx) in iterator:
        x_dev = x.to(device, non_blocking=True)
        ei_dev = ei.to(device, non_blocking=True)
        ea = (edge_attr.to(device, non_blocking=True)
              if edge_attr is not None else None)

        attn_list = model([x_dev], [ei_dev], [None],
                          edge_attr_list=[ea] if ea is not None else None)

        if i == 0 and model.slot_max_cells > 0:
            N = x_dev.shape[0]
            col = model.slot_candidate_snr_column
            if 0 <= col < x_dev.shape[1]:
                prio = x_dev[:, col].detach().cpu().numpy()
                if N > model.slot_max_cells:
                    k = model.slot_max_cells
                    thresh = np.partition(prio, -k)[-k]
                else:
                    thresh = prio.min()
                log(f"🔎 Slot subsample diag: N={N}, "
                    f"max_cells={model.slot_max_cells}, "
                    f"prio column={col}, "
                    f"priority threshold (top-{model.slot_max_cells})={thresh:.3f}")
        attn = attn_list[0].float().cpu().numpy()
        n_cells = attn.shape[1]

        # argmax over slots -> per-cell assignment in [1, K] (0 reserved).
        pred = attn.argmax(axis=0).astype(np.int64) + 1

        truth = None
        if cinfo is not None and 'cell_cluster_index' in cinfo:
            truth = np.asarray(cinfo['cell_cluster_index'], dtype=np.int64)

        slot_usage = np.bincount(pred, minlength=attn.shape[0] + 1)[1:]

        results.append({
            'event_id': event_idx,
            'pred_labels': pred,
            'truth_labels': truth,
            'slot_usage': slot_usage,
            'n_cells': n_cells,
        })

        if max_events is not None and len(results) >= max_events:
            break
        if debug and i >= 4:
            log('Debug: stopping after 5 events')
            break

    return results


# ============================================================================
# TRAINING ORCHESTRATION
# ============================================================================

def train_model_full(model, train_loader, test_loader, test_generator,
                     optimizer, criterion, scaler, device, args, model_name,
                     cluster_info=None, tracker=None, scheduler=None):
    """Full training pipeline for edge classification."""
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, model_name)
    best_model_path = os.path.join(args.save_dir, f"best_{model_name}")
    metrics_path = os.path.splitext(model_path)[0] + "_metrics.pkl"

    best_f1_sum_score = 0.0
    best_epoch = 0
    start_epoch = 1

    metrics = {
        "train_loss": [], "test_loss": [], "train_accuracy": [], "test_accuracy": [],
        "test_macro_recall": [], "test_macro_precision": [], "test_macro_f1": [],
        "test_weighted_recall": [], "test_weighted_precision": [], "test_weighted_f1": [],
        "test_randomness_metric": [], "test_f1_sum_score": [],
        "test_weighted_recall_score": [], "test_weighted_f1_score": [],
        "epoch_times": [],
        "best_accuracy": 0.0, "best_macro_recall": 0.0, "best_macro_precision": 0.0,
        "best_macro_f1": 0.0, "best_weighted_recall": 0.0, "best_weighted_precision": 0.0,
        "best_weighted_f1": 0.0, "best_randomness_metric": 0.0, "best_f1_sum_score": 0.0,
        "best_weighted_recall_score": 0.0, "best_weighted_f1_score": 0.0,
        "best_epoch": 0, "total_time": 0.0, "args": vars(args),
    }
    for c in range(5):
        for m in ['recall', 'precision', 'f1']:
            metrics[f"test_{m}_class_{c}"] = []

    if args.resume:
        chk = find_latest_checkpoint(args.save_dir, model_name)
        if chk:
            ckpt = torch.load(chk[1], map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scaler and 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = chk[0] + 1
            log(f"[Resume] Loaded checkpoint: {chk[1]}")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'rb') as f:
                        metrics.update(pickle.load(f))
                    best_f1_sum_score = metrics.get("best_f1_sum_score", 0.0)
                    best_epoch = metrics.get("best_epoch", 0)
                except:
                    pass

    early_counter = 0
    log(f"\n🚀 Starting training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        train_res = train_epoch(model, train_loader, optimizer, criterion, scaler,
                                device, args.debug, epoch=epoch)
        if scheduler is not None:
            scheduler.step()
        test_res = evaluate(model, test_loader, criterion, device)
        dt = time.perf_counter() - t0

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
        for c in range(5):
            for m in ['recall', 'precision', 'f1']:
                metrics[f"test_{m}_class_{c}"].append(test_res.get(f'{m}_class_{c}', 0.0))

        if test_res["f1_sum_score"] > best_f1_sum_score + 0.01:
            best_f1_sum_score = test_res["f1_sum_score"]
            best_epoch = epoch
            for key in ["accuracy", "macro_recall", "macro_precision", "macro_f1",
                        "weighted_recall", "weighted_precision", "weighted_f1",
                        "randomness_metric", "f1_sum_score",
                        "weighted_recall_score", "weighted_f1_score"]:
                metrics[f"best_{key}"] = test_res[key]
            metrics["best_epoch"] = best_epoch
            early_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        **({"scaler_state_dict": scaler.state_dict()} if scaler else {})},
                       best_model_path)
            log(f"  💾 Best model (F1_Sum={best_f1_sum_score:.2f})")
        else:
            early_counter += 1

        if not args.debug:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       os.path.join(args.save_dir,
                                    f"{os.path.splitext(model_name)[0]}_epoch{epoch}.pt"))

        if args.debug or epoch % 5 == 0:
            log(f"[Epoch {epoch}] {dt:.1f}s F1_Sum={test_res['f1_sum_score']:.2f} "
                f"Best={best_f1_sum_score:.2f}")
        if tracker:
            tracker.log_measurement(f"epoch_{epoch}_complete",
                                    f"F1_Sum={test_res['f1_sum_score']:.2f}")
        if early_counter >= args.patience:
            log(f"[Early Stop] epoch {epoch}")
            break

    metrics["total_time"] = sum(metrics["epoch_times"])

    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model_base = os.path.splitext(model_name)[0]
        parquet_path = os.path.join(args.save_dir, f"results_{model_base}.parquet")

        _, final_metrics = run_inference(
            model, test_generator, device,
            criterion=nn.CrossEntropyLoss(), num_classes=5,
            debug=args.debug, show_progress=True,
            save_path=parquet_path, model_name=model_base)

        if (final_metrics and final_metrics.get("accuracy", 0) > 0
                and sum(final_metrics.get("class_totals", [])) > 0):
            metrics["final_inference_metrics"] = final_metrics
        else:
            log("⚠️ Final inference produced no labeled samples — keeping "
                "training-loop best_* metrics unchanged")

        metrics["num_events_evaluated"] = len(test_generator)
        metrics["parquet_results_path"] = parquet_path
        save_pickle(metrics, metrics_path)
        log(f"📊 Metrics saved: {metrics_path}")

    return metrics, model, best_model_path


def _val_iou_embedding(model, val_generator, device, min_cluster_size,
                       max_events, metric_fn, debug=False,
                       candidate_snr_column=2, hdbscan_n_jobs=-1,
                       cluster_method='hdbscan', cosine_threshold=0.8,
                       split_strict_factor=1.1, split_min_subcluster_size=5,
                       split_min_cluster_size=2, no_hierarchical_split=False,
                       edge_head_score_threshold=0.5):
    """Mean per-truth-cluster IoU on a validation slice (embedding mode)."""
    results = run_embedding_inference(
        model, val_generator, device,
        min_cluster_size=min_cluster_size,
        debug=debug, show_progress=False, max_events=max_events,
        candidate_snr_column=candidate_snr_column,
        hdbscan_n_jobs=hdbscan_n_jobs,
        cluster_method=cluster_method,
        cosine_threshold=cosine_threshold,
        split_strict_factor=split_strict_factor,
        split_min_subcluster_size=split_min_subcluster_size,
        split_min_cluster_size=split_min_cluster_size,
        no_hierarchical_split=no_hierarchical_split,
        edge_head_score_threshold=edge_head_score_threshold)

    ious = []
    for r in results:
        if r['truth_labels'] is None:
            continue
        m = metric_fn(r['pred_labels'], r['truth_labels'])
        v = m.get('mean_iou_per_truth')
        if v is not None and np.isfinite(v):
            ious.append(v)
    return float(np.mean(ious)) if ious else 0.0, len(ious)


def train_embedding_mode(args, model_type, feature_refs, pairs, labels,
                         cells, cluster_info, input_dim, feature_names,
                         device, tracker=None):
    """Full training pipeline for --objective embedding."""
    model_name_used = model_type
    exp_name = args.exp_name or (
        f"{model_name_used}_embedding_h{args.hidden_dim}"
        f"_l{args.layers}_d{args.embed_dim}")
    if args.pretrain:
        exp_name += f"_pretrain_{args.mask_type}_r{args.mask_ratio}"
    if args.multi_scale:
        exp_name += "_multiscale"
    model_filename = f"{exp_name}.pt"

    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, f"best_{model_filename}")
    metrics_path = os.path.splitext(
        os.path.join(args.save_dir, model_filename))[0] + "_metrics.pkl"

    log(f"\n{'='*60}\n🔬 EMBEDDING EXPERIMENT: {exp_name}\n{'='*60}")
    log(f"   Model: {model_name_used.upper()} | Features: {input_dim} | "
        f"Hidden: {args.hidden_dim} | Layers: {args.layers} | "
        f"Embed dim: {args.embed_dim}")
    if args.cluster_method == 'cosine_threshold':
        split_tag = ('NO split pass' if args.no_hierarchical_split
                     else f'split(strict={args.split_strict_factor}, '
                          f'min_sub={args.split_min_subcluster_size}, '
                          f'min_cluster={args.split_min_cluster_size})')
        log(f"   Cluster method: cosine_threshold={args.cosine_threshold} | {split_tag}")
    else:
        log(f"   Cluster method: HDBSCAN (min_cluster_size={args.min_cluster_size})")

    try:
        from corrected_evaluation_utils import compute_cluster_metrics_continuous
    except ImportError as e:
        log(f"❌ Cannot import compute_cluster_metrics_continuous: {e}")
        raise

    model = GraphFoundationModel(
        input_dim, args.hidden_dim, 5, device,
        model_type=model_name_used, num_layers=args.layers,
        num_heads=args.heads, dropout=args.dropout,
        layer_weights=args.layer_weights, softmax_weights=args.softmax_weights,
        norm_type=args.norm, debug=args.debug, pretraining=False,
        feature_names=feature_names, multi_scale=args.multi_scale, objective='embedding',
        embed_dim=args.embed_dim).to(device)

    if args.pretrain:
        pretrained_path = os.path.join(args.save_dir, f"pretrained_{model_filename}")
        if os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location=device, weights_only=True)
            pretrained_dict = ckpt['model_state_dict']
            model_dict = model.state_dict()
            transferred = 0
            for k, v in pretrained_dict.items():
                if (k in model_dict
                        and 'reconstruction_head' not in k
                        and 'projection_head' not in k
                        and 'fc' not in k):
                    model_dict[k] = v
                    transferred += 1
            model.load_state_dict(model_dict)
            log(f"✅ Warm-started encoder from {pretrained_path} "
                f"({transferred} tensors transferred)")
        else:
            log(f"⚠️ --pretrain set but {pretrained_path} not found — "
                f"training embedding model from scratch")

    log(f"   Params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if tracker:
        tracker.log_measurement("embedding_model_created")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_epochs = min(5, max(1, args.epochs // 6))
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[warmup_epochs])

    scaler = (torch.amp.GradScaler('cuda')
              if (args.mixed_precision and args.gpu >= 0 and torch.cuda.is_available())
              else None)
    log(f"✅ {'FP16' if scaler else 'FP32'} | "
        f"LR: cosine+{warmup_epochs}ep warmup | T={args.temperature}")

    train_generator = MultiClassBatchGenerator(
        feature_refs, pairs, labels, cells,
        mode='train', cluster_info_dict=cluster_info,
        debug=args.debug, is_bi_directional=True,
        train_ratio=args.train_ratio,
        chunk_size=args.chunk_size,
        inference_only=False,
        limit_events=args.limit_events)
    test_generator = MultiClassBatchGenerator(
        feature_refs, pairs, labels, cells,
        mode='test', cluster_info_dict=cluster_info,
        debug=args.debug, is_bi_directional=True,
        train_ratio=args.train_ratio,
        chunk_size=args.chunk_size,
        inference_only=False,
        limit_events=args.limit_events)
    train_loader = DataLoader(
        train_generator, batch_size=args.batch_size,
        collate_fn=MultiClassBatchGenerator.collate_data,
        pin_memory=True, num_workers=0)
    if tracker:
        tracker.log_measurement("embedding_loaders_ready")

    start_epoch = 1
    best_val_iou = -float('inf')
    best_epoch = 0
    history = []

    if args.resume:
        chk = find_latest_checkpoint(args.save_dir, f"{exp_name}.pt")
        if chk:
            epoch_num, chk_path = chk
            ckpt = torch.load(chk_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scaler and 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_iou = ckpt.get('best_val_iou', -float('inf'))
            best_epoch = ckpt.get('best_epoch', 0)
            log(f"[Resume] Loaded latest checkpoint {chk_path} "
                f"(epoch {epoch_num}, best_val_iou={best_val_iou:.4f})")

    # Sanity check: does the first event carry cluster info?
    # NOTE: we check feature_refs directly rather than pulling a batch from
    # train_loader. Iterating the chunked IterableDataset to get one sample
    # preloads the entire first chunk (thousands of HDF5 reads), which at
    # chunk_size=args.chunk_size costs ~14 minutes on the hh_bbtt_3000_events dataset.
    # The per-ref 'has_cluster' flag is set during load_features_lazy from
    # the same HDF5 dataset check, so it gives the identical answer in
    # milliseconds.
    has_cluster_first = feature_refs[0].get('has_cluster', False)
    log(f"🔎 Cluster-info sanity (event 0): "
        f"{'present' if has_cluster_first else 'MISSING'}")
    if not has_cluster_first:
        raise RuntimeError(
            "No cluster info in the first event. Contrastive loss "
            "would silently no-op — refusing to train.")

    log(f"\n🚀 Embedding training from epoch {start_epoch} to {args.epochs}...")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()

        train_res = train_epoch_embedding(
            model, train_loader, optimizer, scaler, device,
            temperature=args.temperature, debug=args.debug,
            anchor_chunk_size=args.anchor_chunk_size)
        scheduler.step()
        dt = time.perf_counter() - t0

        # Skip stage-1 validation under edge_head: head is untrained and
        # any IoU would poison best_val_iou -> best_model_path selection.
        val_iou = None
        n_val = 0
        run_val = ((epoch % args.val_every == 0) or (epoch == args.epochs))
        if args.cluster_method == 'edge_head' and args.edge_head_epochs > 0:
            run_val = False

        if run_val:
            val_iou, n_val = _val_iou_embedding(
                model, test_generator, device,
                min_cluster_size=args.min_cluster_size,
                max_events=args.val_events,
                metric_fn=compute_cluster_metrics_continuous,
                debug=args.debug,
                candidate_snr_column=args.candidate_snr_column,
                hdbscan_n_jobs=args.hdbscan_n_jobs,
                cluster_method=args.cluster_method,
                cosine_threshold=args.cosine_threshold,
                split_strict_factor=args.split_strict_factor,
                split_min_subcluster_size=args.split_min_subcluster_size,
                split_min_cluster_size=args.split_min_cluster_size,
                no_hierarchical_split=args.no_hierarchical_split)

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    **({'scaler_state_dict': scaler.state_dict()} if scaler else {}),
                    'train_loss': train_res['loss'],
                    'val_iou': val_iou,
                    'best_epoch': best_epoch,
                    'args': vars(args),
                }, best_model_path)
                log(f"  💾 New best (epoch {epoch}, val_IoU={val_iou:.4f} on {n_val} events)")

        history.append({
            'epoch': epoch, 'train_loss': train_res['loss'],
            'n_events_trained': train_res['n_events'],
            'val_iou': val_iou, 'n_val_events': n_val, 'time': dt,
        })

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **({'scaler_state_dict': scaler.state_dict()} if scaler else {}),
            'train_loss': train_res['loss'],
            'val_iou': val_iou,
            'best_val_iou': best_val_iou,
            'best_epoch': best_epoch,
            'args': vars(args),
        }, os.path.join(args.save_dir, f"{exp_name}_epoch{epoch}.pt"))

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            val_str = f" | val_IoU={val_iou:.4f}" if val_iou is not None else ""
            log(f"[Epoch {epoch}/{args.epochs}] {dt:.1f}s | "
                f"train_loss={train_res['loss']:.6f} | "
                f"events={train_res['n_events']}{val_str}")

        if tracker:
            tracker.log_measurement(
                f"embed_epoch_{epoch}",
                f"loss={train_res['loss']:.4f}"
                + (f" val_IoU={val_iou:.4f}" if val_iou is not None else ""))

    # Under edge_head: stage-1 skipped validation so best_model_path is
    # stale (or absent). Model in memory holds the correct final weights.
    if args.cluster_method == 'edge_head':
        log(f"\nℹ️  --cluster-method edge_head: keeping final stage-1 "
            f"weights in memory (best_model_path not written during stage 1)")
    elif os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        log(f"\n✅ Loaded best checkpoint (epoch {best_epoch}, "
            f"val_IoU={best_val_iou:.4f}) for final inference")

    # ---- Stage 2: edge_score_head ----
    if args.cluster_method == 'edge_head':
        best_epoch = args.epochs
        if args.edge_head_epochs <= 0:
            raise RuntimeError(
                "cluster_method='edge_head' requires --edge-head-epochs > 0.")
        log(f"\n🔧 STAGE 2: training edge_score_head for "
            f"{args.edge_head_epochs} epochs "
            f"({'frozen encoder' if args.freeze_encoder_for_edge_head else 'fine-tuning encoder'})")

        if args.freeze_encoder_for_edge_head:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.edge_score_head.parameters():
                p.requires_grad = True

        edge_optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.edge_head_lr)

        for eh_epoch in range(1, args.edge_head_epochs + 1):
            res = train_epoch_edge_head(
                model, train_loader, edge_optimizer, device,
                debug=args.debug,
                freeze_encoder=args.freeze_encoder_for_edge_head)
            log(f"[EdgeHead {eh_epoch}/{args.edge_head_epochs}] "
                f"loss={res['loss']:.6f} (events={res['n_events']})")

        val_iou_final, n_val_final = _val_iou_embedding(
            model, test_generator, device,
            min_cluster_size=args.min_cluster_size,
            max_events=args.val_events,
            metric_fn=compute_cluster_metrics_continuous,
            debug=args.debug,
            candidate_snr_column=args.candidate_snr_column,
            hdbscan_n_jobs=args.hdbscan_n_jobs,
            cluster_method='edge_head',
            cosine_threshold=args.cosine_threshold,
            split_strict_factor=args.split_strict_factor,
            split_min_subcluster_size=args.split_min_subcluster_size,
            split_min_cluster_size=args.split_min_cluster_size,
            no_hierarchical_split=args.no_hierarchical_split,
            edge_head_score_threshold=args.edge_head_score_threshold)
        log(f"📊 Post-stage-2 val_IoU={val_iou_final:.4f} on {n_val_final} events")

        best_val_iou = val_iou_final
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'val_iou': best_val_iou,
            'best_epoch': best_epoch,
            'args': vars(args),
        }, best_model_path)

    # ---- Final inference & metrics ----
    log(f"\n🔮 Running embedding inference on full test split...")
    results = run_embedding_inference(
        model, test_generator, device,
        min_cluster_size=args.min_cluster_size, debug=args.debug,
        candidate_snr_column=args.candidate_snr_column,
        hdbscan_n_jobs=args.hdbscan_n_jobs,
        cluster_method=args.cluster_method,
        cosine_threshold=args.cosine_threshold,
        split_strict_factor=args.split_strict_factor,
        split_min_subcluster_size=args.split_min_subcluster_size,
        split_min_cluster_size=args.split_min_cluster_size,
        no_hierarchical_split=args.no_hierarchical_split,
        edge_head_score_threshold=args.edge_head_score_threshold)

    per_event_metrics = []
    for r in results:
        if r['truth_labels'] is None:
            continue
        m = compute_cluster_metrics_continuous(r['pred_labels'], r['truth_labels'])
        per_event_metrics.append(m)

    if per_event_metrics:
        def _agg(key):
            vals = [m[key] for m in per_event_metrics
                    if m.get(key) is not None and np.isfinite(m[key])]
            return float(np.mean(vals)) if vals else None

        metrics = {
            'mean_iou_per_truth': _agg('mean_iou_per_truth'),
            'mean_iou_per_pred': _agg('mean_iou_per_pred'),
            'n_events': len(per_event_metrics),
            'best_epoch': best_epoch,
            'best_val_iou': best_val_iou,
            'cluster_method': args.cluster_method,
            'per_event': per_event_metrics,
        }
        log(f"\n📊 Final cluster metrics: "
            f"mean_IoU/truth={metrics['mean_iou_per_truth']:.4f} | "
            f"mean_IoU/pred={metrics['mean_iou_per_pred']:.4f} | "
            f"n_events={metrics['n_events']}")
    else:
        metrics = {'n_events': 0, 'best_epoch': best_epoch,
                   'best_val_iou': best_val_iou,
                   'cluster_method': args.cluster_method}
        log("⚠️ No events had ground-truth cluster labels")

    save_pickle({
        'args': vars(args),
        'history': history,
        'cluster_metrics': metrics,
        'best_epoch': best_epoch,
        'best_val_iou': best_val_iou,
        'per_event_predictions': [
            {'event_id': r['event_id'],
             'pred_labels': r['pred_labels'],
             'truth_labels': r['truth_labels']}
            for r in results
        ],
    }, metrics_path)
    log(f"📊 Embedding metrics saved: {metrics_path}")

    if tracker:
        tracker.log_measurement("embedding_training_complete")
        tracker.print_summary()
        tracker.save_report(f"resource_report_{exp_name}.json")

    return metrics, best_model_path


def train_cluster_slot_mode(args, model_type, feature_refs, pairs, labels,
                            cells, cluster_info, input_dim, feature_names,
                            device, tracker=None):
    """
    Full training pipeline for --objective cluster_slots.

    Uses Hungarian matching against ground-truth clusters for checkpoint
    selection via mean IoU/truth on a validation slice. Same evaluation
    metric as the embedding pipeline, so comparisons are apples-to-apples.
    """
    model_name_used = model_type
    exp_name = args.exp_name or (
        f"{model_name_used}_slots_h{args.hidden_dim}"
        f"_l{args.layers}_k{args.num_slots}")
    if args.multi_scale:
        exp_name += "_multiscale"
    model_filename = f"{exp_name}.pt"

    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, f"best_{model_filename}")
    metrics_path = os.path.splitext(
        os.path.join(args.save_dir, model_filename))[0] + "_metrics.pkl"

    log(f"\n{'='*60}\n🔬 CLUSTER-SLOT EXPERIMENT: {exp_name}\n{'='*60}")
    log(f"   Model: {model_name_used.upper()} | Features: {input_dim} | "
        f"Hidden: {args.hidden_dim} | Layers: {args.layers} | "
        f"K={args.num_slots} | iters={args.slot_iterations} | "
        f"T={args.slot_temperature}")

    try:
        from corrected_evaluation_utils import compute_cluster_metrics_continuous
    except ImportError as e:
        log(f"❌ Cannot import compute_cluster_metrics_continuous: {e}")
        raise

    model = GraphFoundationModel(
        input_dim, args.hidden_dim, 5, device,
        model_type=model_name_used, num_layers=args.layers,
        num_heads=args.heads, dropout=args.dropout,
        layer_weights=args.layer_weights, softmax_weights=args.softmax_weights,
        norm_type=args.norm, debug=args.debug, pretraining=False,
        feature_names=feature_names, multi_scale=args.multi_scale,
        objective='cluster_slots',
        embed_dim=args.embed_dim,
        num_slots=args.num_slots,
        slot_iterations=args.slot_iterations,
        slot_temperature=args.slot_temperature,
        slot_max_cells=args.slot_max_cells,
        slot_candidate_snr_column=args.slot_candidate_snr_column,
        slot_snr_threshold=args.slot_snr_threshold).to(device)

    log(f"   Params: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if tracker:
        tracker.log_measurement("slot_model_created")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_epochs = min(5, max(1, args.epochs // 6))
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[warmup_epochs])

    scaler = (torch.amp.GradScaler('cuda')
              if (args.mixed_precision and args.gpu >= 0 and torch.cuda.is_available())
              else None)
    log(f"✅ {'FP16' if scaler else 'FP32'} | "
        f"LR: cosine+{warmup_epochs}ep warmup | "
        f"Hungarian={'on' if args.slot_hungarian else 'off'}")

    train_generator = MultiClassBatchGenerator(
        feature_refs, pairs, labels, cells,
        mode='train', cluster_info_dict=cluster_info,
        debug=args.debug, is_bi_directional=True,
        train_ratio=args.train_ratio,
        chunk_size=args.chunk_size,
        inference_only=False,
        limit_events=args.limit_events)
    test_generator = MultiClassBatchGenerator(
        feature_refs, pairs, labels, cells,
        mode='test', cluster_info_dict=cluster_info,
        debug=args.debug, is_bi_directional=True,
        train_ratio=args.train_ratio,
        chunk_size=args.chunk_size,
        inference_only=False,
        limit_events=args.limit_events)
    train_loader = DataLoader(
        train_generator, batch_size=args.batch_size,
        collate_fn=MultiClassBatchGenerator.collate_data,
        pin_memory=True, num_workers=0)
    if tracker:
        tracker.log_measurement("slot_loaders_ready")

    start_epoch = 1
    best_val_iou = -float('inf')
    best_epoch = 0
    history = []

    if args.resume:
        chk = find_latest_checkpoint(args.save_dir, f"{exp_name}.pt")
        if chk:
            epoch_num, chk_path = chk
            ckpt = torch.load(chk_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scaler and 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_iou = ckpt.get('best_val_iou', -float('inf'))
            best_epoch = ckpt.get('best_epoch', 0)
            log(f"[Resume] Loaded latest checkpoint {chk_path} "
                f"(epoch {epoch_num}, best_val_iou={best_val_iou:.4f})")

    # Sanity check: does the first event carry cluster info?
    # NOTE: we check feature_refs directly rather than pulling a batch from
    # train_loader. Iterating the chunked IterableDataset to get one sample
    # preloads the entire first chunk (thousands of HDF5 reads), which at
    # chunk_size=args.chunk_size costs ~14 minutes on the hh_bbtt_3000_events dataset.
    # The per-ref 'has_cluster' flag is set during load_features_lazy from
    # the same HDF5 dataset check, so it gives the identical answer in
    # milliseconds.
    has_cluster_first = feature_refs[0].get('has_cluster', False)
    log(f"🔎 Cluster-info sanity (event 0): "
        f"{'present' if has_cluster_first else 'MISSING'}")
    if not has_cluster_first and args.slot_hungarian:
        raise RuntimeError(
            "Hungarian matching requires cluster info, but the first event "
            "does not carry cell_cluster_index. Use --slot-recon-weight > 0 "
            "for unsupervised-only training.")

    log(f"\n🚀 Cluster-slot training from epoch {start_epoch} to {args.epochs}...")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()

        train_res = train_epoch_cluster_slots(
            model, train_loader, optimizer, scaler, device,
            no_object_weight=args.slot_no_object_weight,
            debug=args.debug,
            recon_weight=args.slot_recon_weight)
        scheduler.step()
        dt = time.perf_counter() - t0

        val_iou = None
        n_val = 0
        slot_utilization = None
        if (epoch % args.val_every == 0) or (epoch == args.epochs):
            val_results = run_cluster_slot_inference(
                model, test_generator, device,
                debug=args.debug, show_progress=False,
                max_events=args.val_events)

            ious = []
            per_event_slot_usage = []
            for r in val_results:
                per_event_slot_usage.append(r['slot_usage'])
                if r['truth_labels'] is None:
                    continue
                m = compute_cluster_metrics_continuous(
                    r['pred_labels'], r['truth_labels'])
                v = m.get('mean_iou_per_truth')
                if v is not None and np.isfinite(v):
                    ious.append(v)
            val_iou = float(np.mean(ious)) if ious else 0.0
            n_val = len(ious)

            # Fraction of slots receiving >=1% of cells per event, averaged.
            if per_event_slot_usage:
                usage_fracs = []
                for u in per_event_slot_usage:
                    total = u.sum()
                    if total > 0:
                        usage_fracs.append((u >= 0.01 * total).sum() / len(u))
                slot_utilization = (float(np.mean(usage_fracs))
                                    if usage_fracs else 0.0)

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    **({'scaler_state_dict': scaler.state_dict()} if scaler else {}),
                    'train_loss': train_res['match_loss'],
                    'val_iou': val_iou,
                    'best_epoch': best_epoch,
                    'args': vars(args),
                }, best_model_path)
                slot_str = (f", slot_util={slot_utilization:.2f}"
                            if slot_utilization is not None else "")
                log(f"  💾 New best (epoch {epoch}, val_IoU={val_iou:.4f} "
                    f"on {n_val} events{slot_str})")

        history.append({
            'epoch': epoch,
            'match_loss': train_res['match_loss'],
            'recon_loss': train_res['recon_loss'],
            'n_events_trained': train_res['n_events'],
            'val_iou': val_iou,
            'n_val_events': n_val,
            'slot_utilization': slot_utilization,
            'time': dt,
        })

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **({'scaler_state_dict': scaler.state_dict()} if scaler else {}),
            'train_loss': train_res['match_loss'],
            'val_iou': val_iou,
            'best_val_iou': best_val_iou,
            'best_epoch': best_epoch,
            'args': vars(args),
        }, os.path.join(args.save_dir, f"{exp_name}_epoch{epoch}.pt"))

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            parts = [f"[Epoch {epoch}/{args.epochs}] {dt:.1f}s",
                     f"match_loss={train_res['match_loss']:.4f}"]
            if args.slot_recon_weight > 0:
                parts.append(f"recon_loss={train_res['recon_loss']:.4f}")
            parts.append(f"events={train_res['n_events']}")
            if val_iou is not None:
                parts.append(f"val_IoU={val_iou:.4f}")
            if slot_utilization is not None:
                parts.append(f"slot_util={slot_utilization:.2f}")
            log(" | ".join(parts))

        if tracker:
            tracker.log_measurement(
                f"slot_epoch_{epoch}",
                f"match_loss={train_res['match_loss']:.4f}"
                + (f" val_IoU={val_iou:.4f}" if val_iou is not None else ""))

    # Load best checkpoint for final inference
    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        log(f"\n✅ Loaded best checkpoint (epoch {best_epoch}, "
            f"val_IoU={best_val_iou:.4f}) for final inference")

    log(f"\n🔮 Running cluster-slot inference on full test split...")
    results = run_cluster_slot_inference(
        model, test_generator, device, debug=args.debug, show_progress=True)

    per_event_metrics = []
    all_slot_usage = []
    for r in results:
        all_slot_usage.append(r['slot_usage'])
        if r['truth_labels'] is None:
            continue
        m = compute_cluster_metrics_continuous(r['pred_labels'], r['truth_labels'])
        per_event_metrics.append(m)

    if per_event_metrics:
        def _agg(key):
            vals = [m[key] for m in per_event_metrics
                    if m.get(key) is not None and np.isfinite(m[key])]
            return float(np.mean(vals)) if vals else None

        mean_slot_utilization = None
        if all_slot_usage:
            fracs = []
            for u in all_slot_usage:
                total = u.sum()
                if total > 0:
                    fracs.append((u >= 0.01 * total).sum() / len(u))
            mean_slot_utilization = float(np.mean(fracs)) if fracs else 0.0

        metrics = {
            'mean_iou_per_truth': _agg('mean_iou_per_truth'),
            'mean_iou_per_pred': _agg('mean_iou_per_pred'),
            'n_events': len(per_event_metrics),
            'best_epoch': best_epoch,
            'best_val_iou': best_val_iou,
            'cluster_method': 'cluster_slots',
            'num_slots': args.num_slots,
            'slot_utilization': mean_slot_utilization,
            'per_event': per_event_metrics,
        }
        log(f"\n📊 Final cluster metrics: "
            f"mean_IoU/truth={metrics['mean_iou_per_truth']:.4f} | "
            f"mean_IoU/pred={metrics['mean_iou_per_pred']:.4f} | "
            f"slot_util={mean_slot_utilization:.3f} | "
            f"n_events={metrics['n_events']}")
    else:
        metrics = {'n_events': 0, 'best_epoch': best_epoch,
                   'best_val_iou': best_val_iou,
                   'cluster_method': 'cluster_slots',
                   'num_slots': args.num_slots}
        log("⚠️ No events had ground-truth cluster labels")

    save_pickle({
        'args': vars(args),
        'history': history,
        'cluster_metrics': metrics,
        'best_epoch': best_epoch,
        'best_val_iou': best_val_iou,
        'per_event_predictions': [
            {'event_id': r['event_id'],
             'pred_labels': r['pred_labels'],
             'truth_labels': r['truth_labels'],
             'slot_usage': r['slot_usage']}
            for r in results
        ],
    }, metrics_path)
    log(f"📊 Cluster-slot metrics saved: {metrics_path}")

    if tracker:
        tracker.log_measurement("cluster_slot_training_complete")
        tracker.print_summary()
        tracker.save_report(f"resource_report_{exp_name}.json")

    return metrics, best_model_path


# ============================================================================
# SINGLE MODEL TRAINING (DISPATCH)
# ============================================================================

def train_single_model(args, model_type=None, tracker=None):
    model_name_used = model_type or args.model

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
        log(f"🎯 GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        log("🎯 CPU")
    if tracker:
        tracker.log_measurement("device_set")

    feature_refs, pairs, labels, cluster_info, input_dim, feature_names, cells = \
        load_features_lazy(args.data_dir, args, tracker)

    if args.objective == 'embedding':
        return train_embedding_mode(
            args, model_name_used, feature_refs, pairs, labels,
            cells, cluster_info, input_dim, feature_names,
            device, tracker)

    if args.objective == 'cluster_slots':
        return train_cluster_slot_mode(
            args, model_name_used, feature_refs, pairs, labels,
            cells, cluster_info, input_dim, feature_names,
            device, tracker)

    exp_name = args.exp_name or (
        f"{model_name_used}_"
        f"{'baseline' if not args.all_features else 'all'}_"
        f"h{args.hidden_dim}_l{args.layers}")
    if model_name_used in ['gat', 'transformer']:
        exp_name += f"_heads{args.heads}"
    if args.weighted_loss:
        exp_name += f"_{args.weight_strategy}"
    if args.pretrain:
        exp_name += f"_pretrain_{args.mask_type}_r{args.mask_ratio}"
    if args.multi_scale:
        exp_name += "_multiscale"
    model_filename = f"{exp_name}.pt"

    log(f"\n{'='*60}\n🔬 EXPERIMENT: {exp_name}\n{'='*60}")
    log(f"   Model: {model_name_used.upper()} | Features: {input_dim} | "
        f"Hidden: {args.hidden_dim} | Layers: {args.layers}")
    if args.pretrain:
        log(f"   Pretraining: {args.mask_type} masking | Ratio: {args.mask_ratio} | "
            f"Pretrain epochs: {args.pretrain_epochs}")

    # ---- INFERENCE-ONLY MODE ----
    if args.inference_only:
        log(f"\n{'='*60}\n🔮 INFERENCE-ONLY MODE\n{'='*60}")
        best_model_path = os.path.join(args.save_dir, f"best_{model_filename}")
        if not os.path.exists(best_model_path):
            chk = find_latest_checkpoint(args.save_dir, model_filename)
            if chk:
                best_model_path = chk[1]
            else:
                log("❌ No checkpoint found.")
                return None, None

        ckpt = torch.load(best_model_path, map_location=device, weights_only=True)

        model = GraphFoundationModel(
            input_dim, args.hidden_dim, 5, device, model_name_used, args.layers,
            args.heads, args.dropout, args.layer_weights, args.softmax_weights,
            args.norm, args.debug, pretraining=False,
            feature_names=feature_names, multi_scale=args.multi_scale).to(device)

        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in ckpt['model_state_dict'].items()
            if k in model_dict and 'reconstruction_head' not in k and 'fc' not in k
        }
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
            chunk_size=args.chunk_size, inference_only=False,
            limit_events=args.limit_events)

        _, test_metrics = run_inference(
            model, test_generator, device,
            criterion=nn.CrossEntropyLoss(), num_classes=5,
            debug=args.debug, show_progress=True,
            save_path=parquet_path, model_name=model_base)

        if test_metrics is None:
            log("❌ Inference produced no metrics")
            return None, None

        metrics = {
            'best_epoch': ckpt.get('epoch', '?'),
            'best_accuracy': test_metrics['accuracy'],
            'best_macro_f1': test_metrics['macro_f1'],
            'best_f1_sum_score': test_metrics['f1_sum_score'],
            'best_randomness_metric': test_metrics['randomness_metric'],
            'parquet_results_path': parquet_path,
            'model_path': best_model_path,
            'args': vars(args),
        }
        for c in range(5):
            for m in ['recall', 'precision', 'f1']:
                metrics[f'test_{m}_class_{c}'] = [
                    test_metrics.get(f'{m}_class_{c}', 0.0)]
        save_pickle(metrics, os.path.join(args.save_dir, f"{model_base}_metrics.pkl"))
        log(f"\n✅ INFERENCE COMPLETE! F1_Sum: {test_metrics['f1_sum_score']:.2f} "
            f"| Parquet: {parquet_path}")
        return metrics, best_model_path

    # ---- PRETRAINING MODE ----
    if args.pretrain:
        log(f"\n{'='*60}\n🔧 PRETRAINING MODE\n{'='*60}")

        masking_fn = CalorimeterMasking(
            mask_ratio=args.mask_ratio, mask_type=args.mask_type,
            mask_features=args.mask_features, cells_array=cells,
            cluster_info_dict=cluster_info, geometry_radius=args.geometry_radius)

        pretrain_model = GraphFoundationModel(
            input_dim, args.hidden_dim, 5, device,
            model_type=model_name_used, num_layers=args.layers,
            num_heads=args.heads, dropout=args.dropout,
            layer_weights=args.layer_weights, softmax_weights=args.softmax_weights,
            norm_type=args.norm, debug=args.debug, pretraining=True,
            feature_names=feature_names,
            multi_scale=args.multi_scale).to(device)

        log(f"   Pretrain params: "
            f"{sum(p.numel() for p in pretrain_model.parameters() if p.requires_grad):,}")

        recon_criterion = MaskedReconstructionLoss(
            feature_types=pretrain_model._infer_feature_types(),
            continuous_loss=args.continuous_loss)

        pretrain_optimizer = optim.Adam(pretrain_model.parameters(),
                                        lr=args.lr, weight_decay=args.weight_decay)
        pretrain_scaler = (torch.amp.GradScaler('cuda')
                           if (args.mixed_precision and args.gpu >= 0
                               and torch.cuda.is_available()) else None)
        log(f"✅ Pretraining: {'FP16' if pretrain_scaler else 'FP32'}")

        pretrain_generator = MultiClassBatchGenerator(
            feature_refs, pairs, labels, cells, mode='train',
            cluster_info_dict=cluster_info, debug=args.debug,
            is_bi_directional=True, train_ratio=1.0,
            chunk_size=args.chunk_size, inference_only=False,
            limit_events=args.limit_events)

        pretrain_loader = DataLoader(
            pretrain_generator, batch_size=args.batch_size,
            collate_fn=MultiClassBatchGenerator.collate_data,
            pin_memory=True, num_workers=0)

        pretrained_path = os.path.join(args.save_dir, f"pretrained_{model_filename}")
        pretrain_metrics_path = os.path.join(
            args.save_dir, f"pretrain_metrics_{exp_name}.pkl")

        start_pretrain_epoch = 1
        best_pretrain_loss = float('inf')
        pretrain_metrics_history = []

        if args.resume:
            chk = find_latest_checkpoint(args.save_dir, f"{exp_name}_pretrain.pt")
            if chk:
                epoch_num, chk_path = chk
                ckpt = torch.load(chk_path, map_location=device, weights_only=True)
                pretrain_model.load_state_dict(ckpt['model_state_dict'])
                pretrain_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if pretrain_scaler and 'scaler_state_dict' in ckpt:
                    pretrain_scaler.load_state_dict(ckpt['scaler_state_dict'])
                start_pretrain_epoch = ckpt.get('epoch', 0) + 1
                best_pretrain_loss = ckpt.get(
                    'best_pretrain_loss',
                    ckpt.get('pretrain_loss', float('inf')))
                log(f"[Resume] Loaded latest pretrain checkpoint: {chk_path} "
                    f"(epoch {epoch_num}, best_loss={best_pretrain_loss:.6f})")
                if os.path.exists(pretrain_metrics_path):
                    try:
                        pretrain_metrics_history = load_pickle(pretrain_metrics_path)
                        log(f"[Resume] Loaded {len(pretrain_metrics_history)} "
                            f"prior pretrain epoch records")
                    except Exception as e:
                        log(f"⚠️ Could not load pretrain metrics history: {e}")
                        pretrain_metrics_history = []

            if start_pretrain_epoch > args.pretrain_epochs:
                log(f"✅ Pretraining already complete "
                    f"({start_pretrain_epoch - 1}/{args.pretrain_epochs} epochs done)")

        log(f"\n🚀 Starting pretraining from epoch {start_pretrain_epoch} "
            f"to {args.pretrain_epochs}...")

        for epoch in range(start_pretrain_epoch, args.pretrain_epochs + 1):
            t0 = time.perf_counter()
            pretrain_res = pretrain_epoch(
                pretrain_model, pretrain_loader, pretrain_optimizer,
                recon_criterion, masking_fn, pretrain_scaler, device,
                feature_names, args.debug)
            dt = time.perf_counter() - t0

            pretrain_metrics_history.append({
                'epoch': epoch, 'loss': pretrain_res['loss'],
                'mask_ratio': pretrain_res['mask_ratio'], 'time': dt,
            })

            if pretrain_res['loss'] < best_pretrain_loss:
                best_pretrain_loss = pretrain_res['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': pretrain_model.state_dict(),
                    'optimizer_state_dict': pretrain_optimizer.state_dict(),
                    **({'scaler_state_dict': pretrain_scaler.state_dict()}
                       if pretrain_scaler else {}),
                    'feature_names': feature_names,
                    'input_dim': input_dim,
                    'hidden_dim': args.hidden_dim,
                    'mask_type': args.mask_type,
                    'mask_ratio': args.mask_ratio,
                    'pretrain_loss': pretrain_res['loss'],
                    'best_pretrain_loss': best_pretrain_loss,
                }, pretrained_path)

            torch.save({
                'epoch': epoch,
                'model_state_dict': pretrain_model.state_dict(),
                'optimizer_state_dict': pretrain_optimizer.state_dict(),
                **({'scaler_state_dict': pretrain_scaler.state_dict()}
                   if pretrain_scaler else {}),
                'feature_names': feature_names,
                'input_dim': input_dim,
                'hidden_dim': args.hidden_dim,
                'mask_type': args.mask_type,
                'mask_ratio': args.mask_ratio,
                'pretrain_loss': pretrain_res['loss'],
                'best_pretrain_loss': best_pretrain_loss,
            }, os.path.join(args.save_dir,
                            f"{exp_name}_pretrain_epoch{epoch}.pt"))

            if (epoch == 1 or epoch % max(1, args.pretrain_epochs // 10) == 0
                    or epoch == args.pretrain_epochs):
                log(f"[Pretrain Epoch {epoch}/{args.pretrain_epochs}] {dt:.1f}s | "
                    f"Loss: {pretrain_res['loss']:.6f} | "
                    f"Mask ratio: {pretrain_res['mask_ratio']:.3f}")

            if tracker:
                tracker.log_measurement(f"pretrain_epoch_{epoch}",
                                        f"Loss={pretrain_res['loss']:.4f}")

        save_pickle(pretrain_metrics_history, pretrain_metrics_path)
        log(f"💾 Pretraining complete! Best loss: {best_pretrain_loss:.6f}")

        if not args.finetune_epochs or args.finetune_epochs == 0:
            log("✅ Pretraining only mode - skipping finetuning")
            if tracker:
                tracker.log_measurement("pretraining_complete")
                tracker.print_summary()
                tracker.save_report(f"resource_report_{exp_name}.json")
            return pretrain_metrics_history, pretrained_path

        log(f"\n{'='*60}\n🔧 FINETUNING MODE (from pretrained encoder)\n{'='*60}")

        model = GraphFoundationModel(
            input_dim, args.hidden_dim, 5, device,
            model_type=model_name_used, num_layers=args.layers,
            num_heads=args.heads, dropout=args.dropout,
            layer_weights=args.layer_weights, softmax_weights=args.softmax_weights,
            norm_type=args.norm, debug=args.debug, pretraining=False,
            feature_names=feature_names,
            multi_scale=args.multi_scale).to(device)

        pretrained_dict = pretrain_model.state_dict()
        finetune_dict = model.state_dict()
        transferred_count = 0
        for k, v in pretrained_dict.items():
            if k in finetune_dict and 'reconstruction_head' not in k:
                finetune_dict[k] = v
                transferred_count += 1
        model.load_state_dict(finetune_dict)
        log(f"✅ Transferred {transferred_count} parameters from pretrained encoder")

        args.epochs = args.finetune_epochs

    else:
        model = GraphFoundationModel(
            input_dim, args.hidden_dim, 5, device, model_name_used, args.layers,
            args.heads, args.dropout, args.layer_weights, args.softmax_weights,
            args.norm, args.debug, pretraining=False,
            feature_names=feature_names,
            multi_scale=args.multi_scale).to(device)
        log(f"   Params: "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        if tracker:
            tracker.log_measurement("model_created")

    # ---- COMMON: edge-classification training ----
    all_event_ids = list(range(len(feature_refs)))
    split_idx = int(len(all_event_ids) * args.train_ratio)
    train_labels = labels[all_event_ids[:split_idx]].flatten()

    criterion = create_loss_function(args, train_labels, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_epochs = min(5, max(1, args.epochs // 6))
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[warmup_epochs])

    scaler = (torch.amp.GradScaler('cuda')
              if (args.mixed_precision and args.gpu >= 0 and torch.cuda.is_available())
              else None)
    log(f"✅ {'FP16' if scaler else 'FP32'} | LR: cosine+{warmup_epochs}ep warmup")

    train_generator = MultiClassBatchGenerator(
        feature_refs, pairs, labels, cells, mode='train',
        cluster_info_dict=cluster_info, debug=args.debug,
        is_bi_directional=True, train_ratio=args.train_ratio,
        chunk_size=args.chunk_size, inference_only=False,
        limit_events=args.limit_events)
    test_generator = MultiClassBatchGenerator(
        feature_refs, pairs, labels, cells, mode='test',
        cluster_info_dict=cluster_info, debug=args.debug,
        is_bi_directional=True, train_ratio=args.train_ratio,
        chunk_size=args.chunk_size, inference_only=False,
        limit_events=args.limit_events)
    train_loader = DataLoader(
        train_generator, batch_size=args.batch_size,
        collate_fn=MultiClassBatchGenerator.collate_data,
        pin_memory=True, num_workers=0)
    test_loader = DataLoader(
        test_generator, batch_size=args.batch_size,
        collate_fn=MultiClassBatchGenerator.collate_data,
        pin_memory=True, num_workers=0)
    if tracker:
        tracker.log_measurement("data_loaders_ready")

    metrics, model, model_path = train_model_full(
        model, train_loader, test_loader, test_generator,
        optimizer, criterion, scaler, device, args, model_filename,
        cluster_info, tracker, scheduler=scheduler)

    log(f"\n{'='*60}\n🏁 TRAINING SUMMARY\n{'='*60}")
    log(f"   Best epoch: {metrics['best_epoch']} | "
        f"F1_Sum: {metrics['best_f1_sum_score']:.2f} | "
        f"Accuracy: {metrics['best_accuracy']:.4f}")
    log(f"   Total time: {metrics['total_time']/60:.1f} min")
    if tracker:
        tracker.log_measurement("training_complete")
        tracker.print_summary()
        tracker.save_report(f"resource_report_{exp_name}.json")
    return metrics, model_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    log(f"{'='*70}\n🚀 GRAPH FOUNDATION MODEL (CUDA)\n{'='*70}")
    log(f"Model: {args.model.upper()} | "
        f"Features: {'ALL (42+)' if args.all_features else 'BASELINE (3)'}")
    log(f"Hidden: {args.hidden_dim} | Layers: {args.layers} | GPU: {args.gpu}")

    if args.pretrain and args.objective == 'embedding':
        log(f"Mode: PRETRAINING → EMBEDDING (contrastive)")
    elif args.pretrain and args.objective == 'cluster_slots':
        log(f"Mode: PRETRAINING → CLUSTER-SLOTS")
    elif args.pretrain:
        log(f"Mode: PRETRAINING + "
            f"{'FINETUNING' if args.finetune_epochs > 0 else 'PRETRAINING ONLY'}")
    elif args.inference_only and args.objective == 'embedding':
        log(f"Mode: EMBEDDING INFERENCE ONLY")
    elif args.inference_only and args.objective == 'cluster_slots':
        log(f"Mode: CLUSTER-SLOT INFERENCE ONLY")
    elif args.inference_only:
        log(f"Mode: INFERENCE ONLY")
    elif args.objective == 'embedding':
        log(f"Mode: SUPERVISED CONTRASTIVE EMBEDDINGS")
    elif args.objective == 'cluster_slots':
        log(f"Mode: CLUSTER-SLOT TRANSFORMER")
    else:
        log(f"Mode: SUPERVISED TRAINING")

    if args.objective == 'embedding':
        log(f"Cluster method: {args.cluster_method}")
    if args.objective == 'cluster_slots':
        log(f"Num slots: {args.num_slots} | Iters: {args.slot_iterations} | "
            f"T: {args.slot_temperature}")

    if args.pretrain:
        log(f"Mask: {args.mask_type} (ratio={args.mask_ratio})")

    tracker = ResourceTracker(log_dir=args.save_dir, enabled=args.track_resources)
    if args.track_resources:
        tracker.start()

    if args.analyze_scalability:
        analyze_dataset_scalability(args.data_dir)
        if tracker:
            tracker.save_report("scalability_report.json")
        return

    if args.model == 'all':
        for arch in ['gcn', 'gat', 'transformer', 'sage']:
            log(f"\n{'='*60}\nTraining {arch.upper()}...")
            try:
                import copy
                arch_args = copy.deepcopy(args)
                arch_args.model = arch
                train_single_model(arch_args, arch, tracker)
            except Exception as e:
                log(f"❌ {arch} failed: {e}")
                traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()
    else:
        try:
            train_single_model(args, tracker=tracker)
        except Exception as e:
            log(f"❌ Failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    if tracker:
        tracker.print_summary()
        tracker.save_report("final_resource_report.json")

    log(f"\n{'='*70}\n🎉 PIPELINE COMPLETE!\n{'='*70}")


if __name__ == "__main__":
    main()
