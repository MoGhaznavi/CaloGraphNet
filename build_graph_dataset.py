#!/usr/bin/env python3
"""
COMPREHENSIVE 4-FILE DATASET BUILDER with STRUCTURED CLUSTER INFORMATION
- FIXED: Correct branch mapping (cell_e for energy)
- FIXED: Cluster 0 = NO cluster, valid clusters start at 1
- FIXED: Correct label logic
- NEW: Configurable output naming based on input dataset
"""

import os
import sys
import time
import json
import gc
import math
import psutil
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import awkward as ak
import uproot
import h5py
from sklearn.preprocessing import RobustScaler

# ------------------ PARSE COMMAND LINE ARGUMENTS ------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Process ROOT files into graph ML dataset')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input ROOT file path')
    parser.add_argument('--output-dir', '-o', type=str, default='/storage/mxg1065/processed_data',
                        help='Base output directory (default: /storage/mxg1065/processed_data)')
    parser.add_argument('--dataset-name', '-n', type=str, default=None,
                        help='Custom dataset name for output files (default: derived from input filename)')
    parser.add_argument('--debug', action='store_true', default=True,
                        help='Enable debug output')
    parser.add_argument('--skip-snr-scaling', action='store_true', default=False)
    parser.add_argument('--normalize-energy', action='store_true', default=True)
    parser.add_argument('--energy-normalization', type=str, default='robust',
                        choices=['robust', 'standard'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--chunk-size', type=int, default=50)
    parser.add_argument('--h5-compression', type=str, default='gzip')
    parser.add_argument('--h5-chunk-rows', type=int, default=None)
    return parser.parse_args()

# ------------------ HELPER FUNCTIONS ------------------
def get_dataset_name(input_path, custom_name=None):
    """Generate dataset name from input filename or use custom name"""
    if custom_name:
        return custom_name
    
    # Extract filename without extension
    input_path = Path(input_path)
    base_name = input_path.stem  # Removes extension
    
    # Clean up common patterns
    base_name = base_name.replace('.outputs', '')
    base_name = base_name.replace('xAODAnalysis_', '')
    
    return base_name

def log(msg, debug_mode=True):
    """Conditional logging"""
    print(msg)
    sys.stdout.flush()

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

# ------------------ CELL SIZE AND VOLUME MAPPING ------------------
CELL_SIZES = {
    '0_0': (0.025, 0.1), '0_1': (0.0031, 0.0245), '0_2': (0.025, 0.0245), '0_3': (0.05, 0.0245),
    '0_4': (0.025, 0.1), '0_5': (0.0031, 0.1), '0_6': (0.1, 0.1), '0_7': (0.1, 0.1),
    '1_8': (0.1, 0.09817481), '1_9': (0.1, 0.09817481), '1_10': (0.1, 0.09817481), '1_11': (0.1, 0.09817481),
    '2_21': (0.1, 0.1), '2_22': (0.1, 0.1), '2_23': (0.1, 0.1),
    '3_12': (0.1, 0.09817481), '3_13': (0.1, 0.09817481), '3_14': (0.2, 0.09817481), '3_15': (0.2, 0.09817481),
    '3_16': (0.2, 0.09817481), '3_17': (0.25, 0.09817481), '3_18': (0.1, 0.09817481), '3_19': (0.1, 0.09817481),
    '3_20': (0.2, 0.09817481),
}

SUB_CALO_CATEGORIES = [0, 1, 2, 3]
SUB_CALO_TO_INDEX = {val: idx for idx, val in enumerate(SUB_CALO_CATEGORIES)}
NUM_SUB_CALO = len(SUB_CALO_CATEGORIES)

def get_cell_volume(subCalo, sampling, eta, phi):
    key = f"{subCalo}_{sampling}"
    if key in CELL_SIZES:
        deta, dphi = CELL_SIZES[key]
        r = 1000.0
        volume = deta * dphi * r
        return volume
    return 1.0

# ------------------ MAIN FUNCTION ------------------
def main():
    args = parse_args()
    
    # Setup output directory for this dataset
    dataset_name = get_dataset_name(args.input, args.dataset_name)
    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output file paths with dataset name prefix
    output_files = {
        'pairs': output_dir / f"pairs_{dataset_name}.npy",
        'events': output_dir / f"events_{dataset_name}.h5",
        'labels': output_dir / f"labels_{dataset_name}.npy",
        'cells': output_dir / f"cells_{dataset_name}.npy",
        'metadata': output_dir / f"metadata_{dataset_name}.json"
    }
    
    t_start = time.time()
    
    log("\n" + "="*70)
    log("COMPREHENSIVE 4-FILE DATASET BUILDER (MULTI-DATASET VERSION)")
    log("="*70)
    log(f"Input file:    {args.input}")
    log(f"Dataset name:  {dataset_name}")
    log(f"Output dir:    {output_dir}")
    log(f"Output prefix: {dataset_name}")
    log(f"Chunk size:    {args.chunk_size}")
    log(f"Batch size:    {args.batch_size}")
    log(f"[MEMORY] Initial: {get_memory_usage():.2f} GB\n")
    
    # Save metadata about this processing run
    metadata = {
        'dataset_name': dataset_name,
        'input_file': str(args.input),
        'output_dir': str(output_dir),
        'processing_time': None,
        'num_events': None,
        'num_cells': None,
        'num_pairs': None,
        'args': vars(args)
    }
    
    # ------------------ OPEN ROOT FILE ------------------
    f = uproot.open(args.input)
    possible_trees = ["analysis;1", "analysis", "tree", "CollectionTree", list(f.keys())[0]]
    tree_name = None
    for n in possible_trees:
        if n in f:
            tree_name = n
            break
    if tree_name is None:
        raise RuntimeError("Could not find a candidate tree in the ROOT file.")
    tree = f[tree_name]
    all_branch_names = list(tree.keys())
    log(f"[INFO] Found tree '{tree_name}' with {len(all_branch_names)} branches")
    
    # ============================================================================
    # PHASE 1: EXACT BRANCH MAPPING (FIXED)
    # ============================================================================
    log("\n[PHASE 1] Using EXACT branch names from inspection...")
    
    key_branch_names = {
        "cell_energy": "cell_e",
        "cell_SNR": "cell_SNR",
        "cell_noiseSigma": "cell_noiseSigma",
        "cell_eta": "cell_eta",
        "cell_phi": "cell_phi",
        "cell_cluster_index": "cell_cluster_index",
        "neighbor": "neighbor",
        "cell_subCalo": "cell_subCalo",
        "cell_sampling": "cell_sampling"
    }
    
    # Verify each exists
    for key, name in key_branch_names.items():
        if name not in all_branch_names:
            log(f"  ⚠️  Branch '{name}' not found for key '{key}'")
        else:
            log(f"  ✓ {key}: {name}")
    
    # Read event 0 for structural reference
    log("[PHASE 1] Reading event 0 structural values...")
    entry0 = tree.arrays(list(key_branch_names.values()), entry_start=0, entry_stop=1, library="ak")
    
    def get_event0_field(key):
        br = key_branch_names.get(key)
        if br is None:
            return None
        arr0 = entry0[br][0]
        if arr0 is None:
            return np.array([], dtype=np.float32)
        return ak.to_numpy(arr0)
    
    energy_name = key_branch_names.get("cell_energy")
    noise_name = key_branch_names.get("cell_noiseSigma")
    snr_name = key_branch_names.get("cell_SNR")
    
    if energy_name and noise_name:
        use_energy_noise = True
        log(f"  ✓ Found energy branch: {energy_name}")
        log(f"  ✓ Found noise branch: {noise_name}")
        log(f"  → Will store energy + noise separately")
    elif snr_name:
        use_energy_noise = False
        log(f"  ✓ Found SNR branch: {snr_name}")
        log(f"  → Will store SNR only (fallback mode)")
    else:
        use_energy_noise = False
        log(f"  ⚠️  No energy/noise/SNR branches found - will use zeros")
    
    # Build valid mask from event 0
    if noise_name:
        cell_noise_ev0 = get_event0_field("cell_noiseSigma")
        valid_mask0 = (cell_noise_ev0 != 0) if cell_noise_ev0 is not None else np.ones(1, dtype=bool)
    elif snr_name:
        cell_snr_ev0 = get_event0_field("cell_SNR")
        valid_mask0 = (cell_snr_ev0 != 0)
    else:
        valid_mask0 = np.ones(1, dtype=bool)
    
    num_cells_orig = valid_mask0.size if valid_mask0.size > 0 else 0
    global_valid_indices = np.where(valid_mask0)[0] if valid_mask0.size > 0 else np.array([], dtype=np.int32)
    num_cells_global = len(global_valid_indices)
    global_to_orig = global_valid_indices.astype(np.int32)
    orig_to_global = {int(orig): int(i) for i, orig in enumerate(global_to_orig)}
    log(f"  ✓ {num_cells_global} valid cells (out of {num_cells_orig} original)")
    
    # ------------------ Build connectivity from neighbor branch ------------------
    log("\n[STEP] Building connectivity...")
    neighbor_name = key_branch_names.get("neighbor")
    if neighbor_name is None:
        log("⚠️  neighbor branch not found.")
        pairs_array = np.zeros((0,2), dtype=np.int32)
    else:
        neighbor0_ak = tree.arrays([neighbor_name], entry_start=0, entry_stop=1, library="ak")[neighbor_name][0]
        neighbor0 = ak.to_list(neighbor0_ak)
        pairs = []
        for orig_src, nbrs in enumerate(neighbor0):
            if orig_src not in orig_to_global:
                continue
            src = orig_to_global[orig_src]
            if nbrs is None:
                continue
            for raw in nbrs:
                try:
                    r = int(raw)
                except Exception:
                    continue
                if r in orig_to_global and src != orig_to_global[r]:
                    pairs.append((src, orig_to_global[r]))
        
        log(f"  Raw pairs before deduplication: {len(pairs)}")
        canonical_pairs = []
        for src, dst in pairs:
            if src != dst:
                a, b = min(src, dst), max(src, dst)
                canonical_pairs.append((a, b))
        unique_pairs = list(set(canonical_pairs))
        log(f"  Unique undirected pairs: {len(unique_pairs)}")
        unique_pairs.sort()
        
        pairs_by_source = [[] for _ in range(num_cells_global)]
        for a, b in unique_pairs:
            pairs_by_source[a].append(b)
        
        ordered_pairs = []
        for src in range(num_cells_global):
            if pairs_by_source[src]:
                pairs_by_source[src] = sorted(set(pairs_by_source[src]))
                for dst in pairs_by_source[src]:
                    ordered_pairs.append([src, dst])
        
        pairs_array = np.array(ordered_pairs, dtype=np.int32)
        log(f"  ✓ {pairs_array.shape[0]} edges")
    
    np.save(output_files['pairs'], pairs_array)
    log(f"  ✓ Saved pairs to {output_files['pairs']}")
    
    num_pairs = pairs_array.shape[0] if pairs_array.size else 0
    
    # ------------------ ADD STATIC CELL FEATURES ------------------
    log("\n[STEP] Computing static cell features...")
    
    subcalo_name = key_branch_names.get("cell_subCalo")
    sampling_name = key_branch_names.get("cell_sampling")
    eta_name = key_branch_names.get("cell_eta")
    phi_name = key_branch_names.get("cell_phi")
    
    cell_eta_ev0 = get_event0_field("cell_eta") if eta_name else np.full(num_cells_orig, np.nan)
    cell_phi_ev0 = get_event0_field("cell_phi") if phi_name else np.full(num_cells_orig, np.nan)
    
    if subcalo_name and sampling_name:
        static_data = tree.arrays([subcalo_name, sampling_name], entry_start=0, entry_stop=1, library="ak")
        subcalo_ev0 = ak.to_numpy(static_data[subcalo_name][0])
        sampling_ev0 = ak.to_numpy(static_data[sampling_name][0])
        subcalo_global = subcalo_ev0[global_to_orig]
        sampling_global = sampling_ev0[global_to_orig]
        
        subcalo_onehot = np.zeros((num_cells_global, NUM_SUB_CALO), dtype=np.float32)
        for i, val in enumerate(subcalo_global):
            if val in SUB_CALO_TO_INDEX:
                subcalo_onehot[i, SUB_CALO_TO_INDEX[val]] = 1.0
        
        unique_samplings = np.unique(sampling_global)
        sampling_to_index = {val: idx for idx, val in enumerate(unique_samplings)}
        num_samplings = len(unique_samplings)
        sampling_onehot = np.zeros((num_cells_global, num_samplings), dtype=np.float32)
        for i, val in enumerate(sampling_global):
            if val in sampling_to_index:
                sampling_onehot[i, sampling_to_index[val]] = 1.0
        
        cell_volumes = np.zeros(num_cells_global, dtype=np.float32)
        eta_ev0_filtered = cell_eta_ev0[global_to_orig] if cell_eta_ev0 is not None else np.zeros(num_cells_global)
        phi_ev0_filtered = cell_phi_ev0[global_to_orig] if cell_phi_ev0 is not None else np.zeros(num_cells_global)
        
        for i in range(num_cells_global):
            cell_volumes[i] = get_cell_volume(subcalo_global[i], sampling_global[i], 
                                              eta_ev0_filtered[i], phi_ev0_filtered[i])
        
        cell_deta = np.zeros(num_cells_global, dtype=np.float32)
        cell_dphi = np.zeros(num_cells_global, dtype=np.float32)
        for i in range(num_cells_global):
            key = f"{subcalo_global[i]}_{sampling_global[i]}"
            if key in CELL_SIZES:
                cell_deta[i], cell_dphi[i] = CELL_SIZES[key]
        
        log(f"  ✓ Computed static features for {num_cells_global} cells")
    else:
        log("  ⚠️  Missing subCalo or sampling branches - using defaults")
        subcalo_onehot = np.zeros((num_cells_global, 1), dtype=np.float32)
        sampling_onehot = np.zeros((num_cells_global, 1), dtype=np.float32)
        cell_volumes = np.ones(num_cells_global, dtype=np.float32)
        cell_deta = np.ones(num_cells_global, dtype=np.float32)
        cell_dphi = np.ones(num_cells_global, dtype=np.float32)
        subcalo_global = np.zeros(num_cells_global, dtype=np.int32)
        sampling_global = np.zeros(num_cells_global, dtype=np.int32)
    
    # ------------------ Compute static noise characteristics ------------------
    if use_energy_noise and noise_name:
        log("\n[STEP] Computing per-cell noise statistics...")
        noise_means = np.zeros(num_cells_global, dtype=np.float32)
        noise_stds = np.zeros(num_cells_global, dtype=np.float32)
        noise_counts = np.zeros(num_cells_global, dtype=np.int32)
        
        num_events = tree.num_entries
        for chunk_start in range(0, num_events, args.chunk_size):
            chunk_end = min(chunk_start + args.chunk_size, num_events)
            arrs = tree.arrays([noise_name], entry_start=chunk_start, entry_stop=chunk_end, library="ak")
            noise_chunk_ak = arrs[noise_name]
            try:
                noise_chunk_np = ak.to_numpy(noise_chunk_ak)[:, global_to_orig]
            except:
                noise_py = ak.to_list(noise_chunk_ak)
                noise_chunk_np = np.zeros((chunk_end - chunk_start, num_cells_global), dtype=np.float32)
                for i, row in enumerate(noise_py):
                    r = np.asarray(row).ravel()
                    if len(r) == num_cells_orig:
                        noise_chunk_np[i, :] = r[global_to_orig]
                    else:
                        for j, orig_idx in enumerate(global_to_orig):
                            if orig_idx < len(r):
                                noise_chunk_np[i, j] = r[orig_idx]
            
            for cell_idx in range(num_cells_global):
                cell_noise = noise_chunk_np[:, cell_idx]
                valid = (cell_noise != 0) & ~np.isnan(cell_noise)
                if valid.any():
                    valid_noise = cell_noise[valid]
                    old_count = noise_counts[cell_idx]
                    new_count = old_count + len(valid_noise)
                    if old_count == 0:
                        noise_means[cell_idx] = np.mean(valid_noise)
                        noise_stds[cell_idx] = np.std(valid_noise)
                    else:
                        old_mean = noise_means[cell_idx]
                        new_mean = old_mean + np.sum(valid_noise - old_mean) / new_count
                        old_ssd = noise_stds[cell_idx]**2 * old_count
                        new_ssd = old_ssd + np.sum((valid_noise - old_mean) * (valid_noise - new_mean))
                        noise_means[cell_idx] = new_mean
                        noise_stds[cell_idx] = np.sqrt(new_ssd / new_count)
                    noise_counts[cell_idx] = new_count
            
            del arrs, noise_chunk_np
            gc.collect()
            if args.debug and chunk_start % 500 == 0:
                log(f"    Noise stats: processed events {chunk_start}-{chunk_end-1}")
        
        valid_noise_cells = noise_counts > 0
        noise_categories = np.full(num_cells_global, -1, dtype=np.int8)
        if valid_noise_cells.any():
            noise_percentiles = np.percentile(noise_means[valid_noise_cells], [33, 66])
            for i in range(num_cells_global):
                if noise_counts[i] > 0:
                    if noise_means[i] < noise_percentiles[0]:
                        noise_categories[i] = 0
                    elif noise_means[i] < noise_percentiles[1]:
                        noise_categories[i] = 1
                    else:
                        noise_categories[i] = 2
        
        log(f"  ✓ Computed noise statistics for {num_cells_global} cells")
    else:
        noise_means = np.zeros(num_cells_global, dtype=np.float32)
        noise_stds = np.zeros(num_cells_global, dtype=np.float32)
        noise_counts = np.zeros(num_cells_global, dtype=np.int32)
        noise_categories = np.full(num_cells_global, -1, dtype=np.int8)
    
    # ------------------ Save cells.npy ------------------
    log("\n[STEP] Saving cell metadata")
    
    cells_struct = np.zeros(num_cells_global, dtype=[
        ('global_idx', 'i4'), ('orig_idx', 'i4'),
        ('eta_event0', 'f8'), ('phi_event0', 'f8'),
        ('broken_event0', 'b1'), ('num_neighbors', 'i4'),
        ('subcalo', 'i4'), ('sampling', 'i4'),
        ('deta', 'f4'), ('dphi', 'f4'), ('volume', 'f4'),
        ('noise_mean', 'f4'), ('noise_std', 'f4'), ('noise_count', 'i4'), ('noise_category', 'i1')
    ])
    
    cells_struct['global_idx'] = np.arange(num_cells_global, dtype=np.int32)
    cells_struct['orig_idx'] = global_to_orig
    cells_struct['eta_event0'] = cell_eta_ev0[global_to_orig] if cell_eta_ev0 is not None else np.zeros(num_cells_global)
    cells_struct['phi_event0'] = cell_phi_ev0[global_to_orig] if cell_phi_ev0 is not None else np.zeros(num_cells_global)
    cells_struct['broken_event0'] = (cell_noise_ev0[global_to_orig] == 0) if noise_name else np.zeros(num_cells_global, dtype=bool)
    cells_struct['num_neighbors'] = np.bincount(pairs_array[:,0], minlength=num_cells_global) if num_pairs > 0 else np.zeros(num_cells_global)
    cells_struct['subcalo'] = subcalo_global
    cells_struct['sampling'] = sampling_global
    cells_struct['deta'] = cell_deta
    cells_struct['dphi'] = cell_dphi
    cells_struct['volume'] = cell_volumes
    cells_struct['noise_mean'] = noise_means
    cells_struct['noise_std'] = noise_stds
    cells_struct['noise_count'] = noise_counts
    cells_struct['noise_category'] = noise_categories
    
    np.save(output_files['cells'], cells_struct)
    log(f"  ✓ Saved cells metadata to {output_files['cells']}")
    
    gc.collect()
    log(f"[MEMORY] After connectivity & cells: {get_memory_usage():.2f} GB\n")
    
    # ------------------ PHASE 2: Fit energy scaler ------------------
    if use_energy_noise and args.normalize_energy:
        log("[PHASE 2] Fitting energy scaler...")
        sample_energies = []
        total_vals = 0
        num_events = tree.num_entries
        
        for start in range(0, num_events, args.chunk_size):
            end = min(start + args.chunk_size, num_events)
            arrs = tree.arrays([energy_name], entry_start=start, entry_stop=end, library="ak")
            energy_chunk_ak = arrs[energy_name]
            try:
                energy_np = ak.to_numpy(energy_chunk_ak)[:, global_to_orig]
            except:
                energy_py = ak.to_list(energy_chunk_ak)
                energy_np = np.zeros((end - start, num_cells_global), dtype=np.float32)
                for i, row in enumerate(energy_py):
                    r = np.asarray(row).ravel()
                    if len(r) == num_cells_orig:
                        energy_np[i, :] = r[global_to_orig]
            
            valid_energy = energy_np[energy_np > 0]
            if valid_energy.size > 0:
                if valid_energy.size > 10000:
                    idx = np.random.choice(valid_energy.size, size=10000, replace=False)
                    sample_energies.append(valid_energy[idx])
                    total_vals += 10000
                else:
                    sample_energies.append(valid_energy)
                    total_vals += valid_energy.size
            
            del arrs, energy_np, valid_energy
            gc.collect()
            if args.debug and start % 500 == 0:
                log(f"    Energy scaler: processed events {start}-{end-1}")
        
        if total_vals > 0:
            all_energies = np.concatenate(sample_energies)
            if args.energy_normalization == "robust":
                med = np.median(all_energies)
                q1 = np.percentile(all_energies, 25)
                q3 = np.percentile(all_energies, 75)
                iqr = (q3 - q1) if (q3 - q1) != 0 else 1.0
                energy_center = med
                energy_scale = iqr
            else:
                energy_center = np.mean(all_energies)
                energy_scale = np.std(all_energies)
            log(f"  ✓ Energy scaler: center={energy_center:.6g}, scale={energy_scale:.6g}")
            
            # Save scaler for this dataset
            scaler_info = {'center': energy_center, 'scale': energy_scale, 'method': args.energy_normalization}
            import pickle
            with open(output_dir / f"scaler_{dataset_name}.pkl", 'wb') as f:
                pickle.dump(scaler_info, f)
            
            del all_energies, sample_energies
        else:
            energy_center, energy_scale = 0.0, 1.0
            log("  ⚠️  No valid energies found - using identity scaling")
        gc.collect()
    else:
        energy_center, energy_scale = 0.0, 1.0
    
    # ============================================================================
    # PHASE 3: Process events and write HDF5 + labels.npy (WITH FIXED LABEL LOGIC)
    # ============================================================================
    log("\n[PHASE 3] Processing events and writing HDF5 + labels.npy...")
    
    num_events = tree.num_entries
    metadata['num_events'] = num_events
    metadata['num_cells'] = num_cells_global
    metadata['num_pairs'] = num_pairs
    
    if num_pairs > 0:
        labels_mmap_path = output_dir / f"labels_memmap_{dataset_name}.npy"
        labels_mmap = np.memmap(labels_mmap_path, dtype=np.int8, mode='w+', shape=(num_events, num_pairs))
    else:
        labels_mmap_path = output_dir / f"labels_memmap_{dataset_name}.npy"
        labels_mmap = np.memmap(labels_mmap_path, dtype=np.int8, mode='w+', shape=(num_events, 0))
    
    cluster_name = key_branch_names.get("cell_cluster_index")
    
    with h5py.File(output_files['events'], "w") as h5f:
        
        # Add metadata as attributes
        h5f.attrs['dataset_name'] = dataset_name
        h5f.attrs['input_file'] = str(args.input)
        h5f.attrs['num_events'] = num_events
        h5f.attrs['num_cells'] = num_cells_global
        h5f.attrs['num_pairs'] = num_pairs
        
        if use_energy_noise:
            energy_raw_dset = h5f.create_dataset("cell/energy_raw", shape=(num_events, num_cells_global), dtype=np.float32, compression=args.h5_compression)
            noise_dset = h5f.create_dataset("cell/noise_raw", shape=(num_events, num_cells_global), dtype=np.float32, compression=args.h5_compression)
            snr_computed_dset = h5f.create_dataset("cell/snr_computed", shape=(num_events, num_cells_global), dtype=np.float32, compression=args.h5_compression)
            if snr_name:
                snr_raw_dset = h5f.create_dataset("cell/snr_raw", shape=(num_events, num_cells_global), dtype=np.float32, compression=args.h5_compression)
            if args.normalize_energy:
                energy_norm_dset = h5f.create_dataset("cell/energy_normalized", shape=(num_events, num_cells_global), dtype=np.float32, compression=args.h5_compression)
        else:
            snr_raw_dset = h5f.create_dataset("cell/snr_raw", shape=(num_events, num_cells_global), dtype=np.float32, compression=args.h5_compression)
        
        cluster_dset = h5f.create_dataset("cell/cell_cluster_index", shape=(num_events, num_cells_global), dtype=np.int32, compression=args.h5_compression)
        labels_h5 = h5f.create_dataset("labels", shape=(num_events, num_pairs), dtype=np.int8, compression=args.h5_compression)
        
        for chunk_start in range(0, num_events, args.chunk_size):
            chunk_end = min(chunk_start + args.chunk_size, num_events)
            chunk_size_actual = chunk_end - chunk_start
            log(f"  Processing events {chunk_start}-{chunk_end-1}")
            
            to_read = []
            if use_energy_noise:
                if energy_name: to_read.append(energy_name)
                if noise_name: to_read.append(noise_name)
                if snr_name: to_read.append(snr_name)
            else:
                if snr_name: to_read.append(snr_name)
            if cluster_name:
                to_read.append(cluster_name)
            
            arrs = tree.arrays(to_read, entry_start=chunk_start, entry_stop=chunk_end, library="ak")
            
            # SNR (ROOT)
            if snr_name and snr_name in arrs.fields:
                try:
                    snr_np = ak.to_numpy(arrs[snr_name])[:, global_to_orig]
                except:
                    snr_np = np.zeros((chunk_size_actual, num_cells_global), dtype=np.float32)
            else:
                snr_np = None
            
            if use_energy_noise:
                if energy_name:
                    try:
                        energy_np = ak.to_numpy(arrs[energy_name])[:, global_to_orig]
                    except:
                        energy_np = np.zeros((chunk_size_actual, num_cells_global), dtype=np.float32)
                else:
                    energy_np = np.zeros((chunk_size_actual, num_cells_global), dtype=np.float32)
                
                if noise_name:
                    try:
                        noise_np = ak.to_numpy(arrs[noise_name])[:, global_to_orig]
                    except:
                        noise_np = np.ones((chunk_size_actual, num_cells_global), dtype=np.float32)
                else:
                    noise_np = np.ones((chunk_size_actual, num_cells_global), dtype=np.float32)
                
                energy_raw_dset[chunk_start:chunk_end] = energy_np
                noise_dset[chunk_start:chunk_end] = noise_np
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    snr_computed = np.divide(energy_np, noise_np, where=(noise_np != 0))
                    snr_computed[~np.isfinite(snr_computed)] = 0
                snr_computed_dset[chunk_start:chunk_end] = snr_computed
                
                if snr_name and snr_np is not None:
                    snr_raw_dset[chunk_start:chunk_end] = snr_np
                
                if args.normalize_energy:
                    energy_norm = (energy_np - energy_center) / energy_scale
                    energy_norm_dset[chunk_start:chunk_end] = energy_norm
            else:
                if snr_np is not None:
                    snr_raw_dset[chunk_start:chunk_end] = snr_np
                else:
                    snr_raw_dset[chunk_start:chunk_end] = 0
            
            # Clusters
            if cluster_name:
                try:
                    cluster_np = ak.to_numpy(arrs[cluster_name])[:, global_to_orig]
                except:
                    cluster_np = np.zeros((chunk_size_actual, num_cells_global), dtype=np.int32)
            else:
                cluster_np = np.zeros((chunk_size_actual, num_cells_global), dtype=np.int32)
            
            cluster_dset[chunk_start:chunk_end] = cluster_np
            
            # ================================================================
            # LABELS (FIXED: Cluster 0 = NO cluster, valid clusters start at 1)
            # ================================================================
            if num_pairs > 0:
                labels_chunk = np.zeros((chunk_size_actual, num_pairs), dtype=np.int8)
                
                for i in range(chunk_size_actual):
                    c0 = cluster_np[i, pairs_array[:, 0]]
                    c1 = cluster_np[i, pairs_array[:, 1]]
                    
                    c0_in = (c0 >= 1)
                    c1_in = (c1 >= 1)
                    both_lone = (c0 == 0) & (c1 == 0)
                    same_cluster = (c0 == c1) & c0_in & c1_in
                    diff_cluster = (c0 != c1) & c0_in & c1_in
                    src_only = c0_in & (c1 == 0)
                    dst_only = (c0 == 0) & c1_in
                    
                    lab = np.zeros(num_pairs, dtype=np.int8)
                    lab[both_lone] = 0
                    lab[same_cluster] = 1
                    lab[src_only] = 2
                    lab[dst_only] = 3
                    lab[diff_cluster] = 4
                    
                    labels_chunk[i] = lab
                    
                    # ============ VERIFICATION FOR FIRST EVENT ============
                    actual_event_idx = chunk_start + i
                    if actual_event_idx == 0 and args.debug:
                        print("\n" + "="*70)
                        print(f"🔍 LABEL ALIGNMENT VERIFICATION (Event 0) - {dataset_name}")
                        print("="*70)
                        print(f"   Cluster 0 = NO cluster, Valid clusters start at 1")
                        print(f"\n   First 20 edges:")
                        print(f"   {'Idx':<5} {'Src':<8} {'Dst':<8} {'c0':<4} {'c1':<4} {'Label':<6} {'Match':<6}")
                        print(f"   {'-'*50}")
                        
                        matches = 0
                        for j in range(min(20, num_pairs)):
                            src = pairs_array[j, 0]
                            dst = pairs_array[j, 1]
                            label = lab[j]
                            src_c = c0[j]
                            dst_c = c1[j]
                            
                            if src_c >= 1 and dst_c >= 1:
                                expected = 1 if src_c == dst_c else 4
                            elif src_c >= 1 and dst_c == 0:
                                expected = 2
                            elif src_c == 0 and dst_c >= 1:
                                expected = 3
                            else:
                                expected = 0
                            
                            match = (label == expected)
                            if match:
                                matches += 1
                            match_str = "✅" if match else "❌"
                            print(f"   {j:<5} {src:<8} {dst:<8} {src_c:<4} {dst_c:<4} {label:<6} {match_str}")
                        
                        print(f"\n   Match rate: {matches}/20 ({matches/20*100:.1f}%)")
                        
                        if matches == 20:
                            print(f"   ✅ PERFECT ALIGNMENT!")
                        else:
                            print(f"   🚨 WARNING: Mismatch detected!")
                        
                        unique, counts = np.unique(lab, return_counts=True)
                        print(f"\n   Label distribution (Event 0):")
                        for u, c in zip(unique, counts):
                            pct = c / num_pairs * 100
                            print(f"      Class {u}: {c:,} ({pct:.2f}%)")
                        print("="*70 + "\n")
                
                labels_h5[chunk_start:chunk_end] = labels_chunk
                labels_mmap[chunk_start:chunk_end] = labels_chunk
            
            # Cleanup
            del arrs
            for var in ['energy_np', 'noise_np', 'snr_computed', 'snr_np', 'energy_norm', 'cluster_np', 'labels_chunk']:
                if var in locals():
                    del locals()[var]
            gc.collect()
            
            if args.debug:
                log(f"    [MEMORY] Current: {get_memory_usage():.2f} GB")
    
    log(f"  ✓ Processed all {num_events} events")
    
    # ------------------ FINALIZE LABELS ------------------
    log("\n[STEP] Finalizing labels...")
    labels_mmap.flush()
    labels_mmap._mmap.close()
    
    if os.path.exists(labels_mmap_path):
        if num_pairs > 0:
            final_labels = np.memmap(labels_mmap_path, dtype=np.int8, mode='r', shape=(num_events, num_pairs))
            np.save(output_files['labels'], final_labels)
            del final_labels
        else:
            np.save(output_files['labels'], np.zeros((num_events, 0), dtype=np.int8))
        os.remove(labels_mmap_path)
    
    log(f"  ✓ Saved labels to {output_files['labels']}")
    
    # ------------------ Save metadata ------------------
    metadata['processing_time'] = time.time() - t_start
    with open(output_files['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # ------------------ Final summary ------------------
    log("\n" + "="*70)
    log("✅ DATASET BUILD COMPLETE!")
    log("="*70)
    log(f"📁 Dataset: {dataset_name}")
    log(f"📂 Output directory: {output_dir}")
    log(f"📄 Output files:")
    for name, path in output_files.items():
        if path.exists():
            size_gb = path.stat().st_size / 1024**3
            log(f"    {name}: {path.name} ({size_gb:.2f} GB)")
    log("")
    log(f"📊 Statistics:")
    log(f"    Events: {num_events:,}")
    log(f"    Cells: {num_cells_global:,}")
    log(f"    Edges: {num_pairs:,}")
    log(f"⏱️  Processing time: {metadata['processing_time']:.1f} seconds")
    log("="*70)

if __name__ == "__main__":
    main()
