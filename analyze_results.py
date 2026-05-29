```python
#!/usr/bin/env python3
"""
analyze_results.py - Comprehensive Model Analysis for CaloGraphNet

Generates ROC curves, PR curves, confusion matrices, comparison plots,
and HTML reports for trained GNN models.

Primary metric: F1 Sum Score (FSS) - balances precision AND recall.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import glob
import gc
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from math import pi
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CLASS_NAMES = {
    0: 'Lone-Lone (Noise)',
    1: 'True-True (Same Cluster)',
    2: 'Cluster-Lone (Source Only)',
    3: 'Lone-Cluster (Dest Only)',
    4: 'Cluster-Cluster (Different)'
}
CLASS_COLORS = ['#2c3e50', '#27ae60', '#e74c3c', '#e67e22', '#9b59b6']

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom colormaps
GREEN_COLORMAP = LinearSegmentedColormap.from_list("green_grad", ["#e8f5e9", "#1b5e20"])
BLUE_COLORMAP = LinearSegmentedColormap.from_list("blue_grad", ["#e3f2fd", "#0d47a1"])


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze trained CaloGraphNet models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--models-dir', type=str, required=True,
                        help='Directory containing model pickle files (*_metrics.pkl)')
    parser.add_argument('--parquet-dir', type=str, required=True,
                        help='Directory containing parquet prediction files')
    parser.add_argument('--output-dir', type=str, default='./analysis_output',
                        help='Output directory for results (default: ./analysis_output)')
    parser.add_argument('--max-rows-roc', type=int, default=500000,
                        help='Max rows for ROC computation (default: 500000)')
    parser.add_argument('--max-rows-confusion', type=int, default=1000000,
                        help='Max rows for confusion matrix (default: 1000000)')
    parser.add_argument('--batch-size', type=int, default=100000,
                        help='Batch size for parquet reading (default: 100000)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top models to show in radar chart (default: 5)')
    
    return parser.parse_args()


def create_output_structure(output_dir):
    """Create organized folder structure for outputs."""
    folders = {
        'roc_curves': os.path.join(output_dir, 'figures', 'roc_curves'),
        'pr_curves': os.path.join(output_dir, 'figures', 'pr_curves'),
        'confusion_matrices': os.path.join(output_dir, 'figures', 'confusion_matrices'),
        'comparison_plots': os.path.join(output_dir, 'figures', 'comparison_plots'),
        'metrics_radar': os.path.join(output_dir, 'figures', 'metrics_radar'),
        'tables': os.path.join(output_dir, 'tables'),
        'reports': os.path.join(output_dir, 'reports'),
        'data': os.path.join(output_dir, 'data'),
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    return folders


def extract_model_metrics(metrics):
    """
    Extract comprehensive metrics from training output.
    
    Priority order for model selection:
    1. F1 Sum Score (FSS) - accounts for both FP and FN [BEST]
    2. Weighted F1 Score (WFSS) - accounts for class imbalance
    3. Recall Sum Score (RSS) - original metric (FN only)
    """
    extracted = {
        # Primary metrics (F1-based)
        'f1_sum_score': metrics.get('best_f1_sum_score', 0.0),
        'weighted_f1_score': metrics.get('best_weighted_f1_score', 0.0),
        'macro_f1': metrics.get('best_macro_f1', 0.0),
        'weighted_f1': metrics.get('best_weighted_f1', 0.0),
        
        # Recall-based metrics
        'randomness_metric': metrics.get('best_randomness_metric', 0.0),  # RSS
        'macro_recall': metrics.get('best_macro_recall', 0.0),
        'weighted_recall': metrics.get('best_weighted_recall', 0.0),
        'weighted_recall_score': metrics.get('best_weighted_recall_score', 0.0),
        
        # Precision metrics
        'macro_precision': metrics.get('best_macro_precision', 0.0),
        'weighted_precision': metrics.get('best_weighted_precision', 0.0),
        
        # Accuracy
        'accuracy': metrics.get('best_accuracy', 0.0),
        
        # Training info
        'best_epoch': metrics.get('best_epoch', '?'),
        
        # Per-class metrics
        'per_class_recall': {},
        'per_class_precision': {},
        'per_class_f1': {},
    }
    
    # Extract per-class metrics
    for c in range(5):
        # Recall
        for key in [f'best_recall_class_{c}', f'test_recall_class_{c}']:
            if key in metrics:
                val = metrics[key]
                extracted['per_class_recall'][c] = max(val) if isinstance(val, list) else val
                break
        else:
            extracted['per_class_recall'][c] = 0.0
        
        # Precision
        for key in [f'best_precision_class_{c}', f'test_precision_class_{c}']:
            if key in metrics:
                val = metrics[key]
                extracted['per_class_precision'][c] = max(val) if isinstance(val, list) else val
                break
        else:
            extracted['per_class_precision'][c] = 0.0
        
        # F1
        for key in [f'best_f1_class_{c}', f'test_f1_class_{c}']:
            if key in metrics:
                val = metrics[key]
                extracted['per_class_f1'][c] = max(val) if isinstance(val, list) else val
                break
        else:
            extracted['per_class_f1'][c] = 0.0
    
    # Fallback: compute FSS from RSS if needed
    if extracted['f1_sum_score'] == 0.0 and extracted['randomness_metric'] > 0:
        extracted['f1_sum_score'] = extracted['randomness_metric']
        extracted['f1_is_fallback'] = True
    else:
        extracted['f1_is_fallback'] = False
    
    return extracted


def compute_roc_data(parquet_dir, model_name, max_rows, batch_size):
    """Compute ROC curve data using PyArrow batch iteration."""
    parquet_path = os.path.join(parquet_dir, f"results_{model_name}.parquet")
    if not os.path.exists(parquet_path):
        print(f"      WARNING: Parquet file not found: {parquet_path}")
        return None, None
    
    parquet_file = pq.ParquetFile(parquet_path)
    
    y_true_all = []
    y_scores_all = []
    total_rows = 0
    
    score_cols = ['score_class_0', 'score_class_1', 'score_class_2', 'score_class_3', 'score_class_4']
    
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=['true_label'] + score_cols
    ):
        chunk = batch.to_pandas()
        y_true_all.append(chunk['true_label'].values)
        y_scores_all.append(chunk[score_cols].values)
        total_rows += len(chunk)
        if total_rows >= max_rows:
            break
    
    if not y_true_all:
        return None, None
    
    y_true = np.concatenate(y_true_all)[:max_rows]
    y_scores = np.concatenate(y_scores_all)[:max_rows]
    return y_true, y_scores


def compute_confusion_matrix(parquet_dir, model_name, max_rows, batch_size):
    """Compute confusion matrix using PyArrow batch iteration."""
    parquet_path = os.path.join(parquet_dir, f"results_{model_name}.parquet")
    if not os.path.exists(parquet_path):
        return None
    
    parquet_file = pq.ParquetFile(parquet_path)
    cm = np.zeros((5, 5), dtype=np.int64)
    total_processed = 0
    
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=['true_label', 'pred_label']
    ):
        chunk = batch.to_pandas()
        y_true = chunk['true_label'].values
        y_pred = chunk['pred_label'].values
        
        for i in range(5):
            for j in range(5):
                cm[i, j] += np.sum((y_true == i) & (y_pred == j))
        
        total_processed += len(chunk)
        if total_processed >= max_rows:
            break
    
    return cm


def plot_roc_curves(model_name, extracted, folders, class_names, class_colors):
    """Generate ROC curves with F1 annotation."""
    print(f"      Computing ROC curves...")
    y_true, y_scores = compute_roc_data(args.parquet_dir, model_name, args.max_rows_roc, args.batch_size)
    
    if y_true is None:
        print("      WARNING: Skipping ROC (no data)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # ROC curves
    ax = axes[0]
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    
    for i in range(5):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=class_colors[i], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # F1 vs Recall scatter
    ax2 = axes[1]
    f1_scores = [extracted['per_class_f1'].get(c, 0) for c in range(5)]
    recall_scores = [extracted['per_class_recall'].get(c, 0) for c in range(5)]
    precision_scores = [extracted['per_class_precision'].get(c, 0) for c in range(5)]
    
    sizes = [max(50, 200 * p) for p in precision_scores]
    
    for i in range(5):
        ax2.scatter(recall_scores[i], f1_scores[i], s=sizes[i], 
                   color=class_colors[i], alpha=0.7, edgecolors='black', linewidth=1.5)
        ax2.annotate(class_names[i].split(' (')[0], 
                    (recall_scores[i], f1_scores[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title(f'Precision-Recall Tradeoff\n(Bubble size = Precision, FSS={extracted["f1_sum_score"]:.2f})', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folders['roc_curves'], f"{model_name}_roc_f1.png"), dpi=150, bbox_inches='tight')
    plt.close()
    del y_true, y_scores, y_true_bin
    gc.collect()
    print(f"   Done: ROC + F1 tradeoff plot saved")


def plot_pr_curves(model_name, extracted, folders, class_names, class_colors, max_rows, batch_size):
    """Generate Precision-Recall curves for each class."""
    print(f"      Computing Precision-Recall curves...")
    y_true, y_scores = compute_roc_data(args.parquet_dir, model_name, max_rows, batch_size)
    
    if y_true is None:
        print("      WARNING: Skipping PR curves (no data)")
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
        
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        
        ax.plot(recall, precision, color=class_colors[i], lw=2, label=f'AP={ap:.3f}')
        
        # F1 iso-lines
        for f1_level in [0.2, 0.4, 0.6, 0.8]:
            x = np.linspace(0.01, 1, 100)
            y = f1_level * x / (2 * x - f1_level)
            y = np.clip(y, 0, 1)
            valid = (2 * x - f1_level) > 0
            if valid.any():
                ax.plot(x[valid], y[valid], 'k--', alpha=0.2, linewidth=0.5)
        
        ax.plot(recall[best_f1_idx], precision[best_f1_idx], 'o', 
                color=class_colors[i], markersize=8, markeredgecolor='black', markeredgewidth=1.5)
        
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title(f'{class_names[i]}\nAP={ap:.3f} | Best F1={f1_scores[best_f1_idx]:.3f}', 
                    fontsize=11, fontweight='bold')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
    
    # Summary plot
    ax_summary = axes[5]
    for i in range(5):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        ax_summary.plot(recall, precision, color=class_colors[i], lw=2, alpha=0.7,
                       label=f'{class_names[i].split(" (")[0]} (AP={all_aps[i]:.3f})')
    
    ax_summary.set_xlabel('Recall', fontsize=12)
    ax_summary.set_ylabel('Precision', fontsize=12)
    ax_summary.set_title(f'All Classes - {model_name}\nMean AP: {np.mean(all_aps):.3f} | FSS: {extracted["f1_sum_score"]:.2f}', 
                        fontsize=12, fontweight='bold')
    ax_summary.legend(loc='lower left', fontsize=7)
    ax_summary.grid(True, alpha=0.3)
    
    plt.suptitle(f'Precision-Recall Curves - {model_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(folders['pr_curves'], f"{model_name}_pr_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    del y_true, y_scores, y_true_bin
    gc.collect()
    print(f"   Done: PR curves saved (Mean AP: {np.mean(all_aps):.3f})")


def plot_confusion_matrix_enhanced(model_name, extracted, folders, class_names, class_colors, max_rows, batch_size):
    """Generate confusion matrix with F1 and Precision annotations."""
    print(f"      Computing confusion matrix...")
    cm = compute_confusion_matrix(args.parquet_dir, model_name, max_rows, batch_size)
    
    if cm is None:
        print("      WARNING: Skipping confusion matrix (no data)")
        return
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Confusion matrix
    ax = axes[0]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=[class_names[i] for i in range(5)],
                yticklabels=[class_names[i] for i in range(5)],
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Proportion'})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    # Per-class metrics bar chart
    ax2 = axes[1]
    x = np.arange(5)
    width = 0.25
    
    recall_vals = [extracted['per_class_recall'].get(c, 0) for c in range(5)]
    precision_vals = [extracted['per_class_precision'].get(c, 0) for c in range(5)]
    f1_vals = [extracted['per_class_f1'].get(c, 0) for c in range(5)]
    
    ax2.bar(x - width, recall_vals, width, label='Recall', color='#3498db', alpha=0.8)
    ax2.bar(x, precision_vals, width, label='Precision', color='#2ecc71', alpha=0.8)
    ax2.bar(x + width, f1_vals, width, label='F1', color='#e74c3c', alpha=0.8)
    
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title(f'Per-Class Performance\nFSS={extracted["f1_sum_score"]:.2f}', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([class_names[i].split(' (')[0] for i in range(5)], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(folders['confusion_matrices'], f"{model_name}_confusion_enhanced.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Done: Enhanced confusion matrix saved")


def plot_radar_chart(all_results, folders, top_n):
    """Create radar chart comparing top models across multiple metrics."""
    top_models = sorted(all_results, key=lambda x: x['f1_sum_score'], reverse=True)[:top_n]
    
    if not top_models:
        return
    
    metrics_to_plot = ['F1 Score', 'Macro Recall', 'Macro Precision', 'Weighted F1', 'Accuracy']
    N = len(metrics_to_plot)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    colors = plt.cm.Set1(np.linspace(0, 1, len(top_models)))
    
    for idx, model in enumerate(top_models):
        values = [
            model['f1_sum_score'] / 5.0,
            model['macro_recall'],
            model['macro_precision'],
            model['weighted_f1'],
            model['accuracy']
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model['name'][:25], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Model Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(folders['metrics_radar'], "metrics_radar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Done: Radar chart saved")


def plot_metric_comparison(all_results, folders, class_names):
    """Create comprehensive comparison of F1 and Recall metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    all_results_sorted = sorted(all_results, key=lambda x: x['f1_sum_score'], reverse=True)
    names = [r['name'] for r in all_results_sorted]
    
    # F1 Sum Score vs Recall Sum Score
    ax = axes[0, 0]
    x = np.arange(len(names))
    width = 0.35
    
    f1_scores = [r['f1_sum_score'] for r in all_results_sorted]
    recall_scores = [r['randomness_metric'] for r in all_results_sorted]
    
    ax.bar(x - width/2, f1_scores, width, label='F1 Sum Score (FSS) [BEST]', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, recall_scores, width, label='Recall Sum Score (RSS)', color='#3498db', alpha=0.8)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Random Baseline (1.0)')
    ax.axhline(y=5.0, color='green', linestyle='--', alpha=0.3, label='Perfect (5.0)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Score (1.0=random, 5.0=perfect)', fontsize=12)
    ax.set_title('FSS vs RSS Comparison\n(FSS accounts for False Positives!)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # F1 vs Recall scatter
    ax = axes[0, 1]
    for r in all_results_sorted:
        color = '#e74c3c' if r['f1_sum_score'] > r['randomness_metric'] else '#3498db'
        ax.scatter(r['randomness_metric'], r['f1_sum_score'], s=100, color=color, alpha=0.6)
        ax.annotate(r['name'][:15], (r['randomness_metric'], r['f1_sum_score']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=7)
    
    ax.plot([1, 5], [1, 5], 'k--', alpha=0.3)
    ax.set_xlabel('Recall Sum Score (RSS)', fontsize=12)
    ax.set_ylabel('F1 Sum Score (FSS)', fontsize=12)
    ax.set_title('FSS vs RSS Scatter Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Per-class F1 scores (top 7)
    ax = axes[1, 0]
    top_7 = all_results_sorted[:7]
    x = np.arange(5)
    width = 0.12
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    
    for i, model in enumerate(top_7):
        f1_vals = [model['per_class_f1'].get(c, 0) for c in range(5)]
        offset = width * (i - 3)
        ax.bar(x + offset, f1_vals, width, label=model['name'][:20], alpha=0.8, color=colors[i])
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Scores (Top 7 Models)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([class_names[i].split(' (')[0] for i in range(5)], rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Model architecture comparison
    ax = axes[1, 1]
    arch_data = {}
    for r in all_results_sorted:
        arch = r.get('architecture', 'unknown')
        if arch not in arch_data:
            arch_data[arch] = []
        arch_data[arch].append(r['f1_sum_score'])
    
    arch_names = list(arch_data.keys())
    arch_means = [np.mean(arch_data[a]) for a in arch_names]
    arch_stds = [np.std(arch_data[a]) for a in arch_names]
    
    x = np.arange(len(arch_names))
    bar_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(arch_names)]
    bars = ax.bar(x, arch_means, yerr=arch_stds, capsize=5, color=bar_colors)
    
    for bar, mean in zip(bars, arch_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
               f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in arch_names])
    ax.set_ylabel('Mean F1 Sum Score', fontsize=12)
    ax.set_title('Architecture Performance Comparison', fontsize=14, fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folders['comparison_plots'], "comprehensive_metrics.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Done: Comprehensive metric comparison saved")


def generate_html_report(all_results, df_display, folders, class_names, datetime_str):
    """Generate comprehensive HTML report."""
    top_model = df_display.iloc[0]
    
    html_report = f"""<!DOCTYPE html>
<html>
<head>
    <title>CaloGraphNet - Comprehensive Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .main-container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: white; text-align: center; font-size: 3em; margin-bottom: 10px; }}
        .subtitle {{ text-align: center; color: rgba(255,255,255,0.9); margin-bottom: 40px; }}
        .container {{ background-color: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.2); margin-bottom: 30px; }}
        h2 {{ color: #2c3e50; border-left: 5px solid #3498db; padding-left: 15px; margin-top: 0; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-box {{ padding: 25px; border-radius: 12px; color: white; text-align: center; }}
        .stat-box.f1 {{ background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: #1a5e20; }}
        .stat-box.macro {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: #1a237e; }}
        .stat-box.accuracy {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: #4a1a1a; }}
        .stat-number {{ font-size: 48px; font-weight: bold; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; margin-top: 5px; }}
        .metric-explanation {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px; text-align: center; }}
        td {{ padding: 10px; text-align: center; border-bottom: 1px solid #e0e0e0; }}
        tr:hover {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="main-container">
        <h1>CaloGraphNet Analysis</h1>
        <p class="subtitle">Comprehensive Performance Report | {datetime_str}</p>
        
        <div class="container">
            <h2>Primary Metric: F1 Sum Score (FSS)</h2>
            <div class="metric-explanation">
                <strong>Why F1 Sum Score?</strong><br>
                FSS = Σ(F1 per class) [range: 1.0 (random) → 5.0 (perfect)]<br>
                - Accounts for BOTH False Positives AND False Negatives<br>
                - More conservative than Recall Sum Score (RSS)<br>
                - F1 = 2 × (Precision × Recall) / (Precision + Recall)
            </div>
            
            <div class="stats">
                <div class="stat-box f1">
                    <div class="stat-number">{top_model['F1 Sum Score']:.2f}</div>
                    <div class="stat-label">Best F1 Sum Score [Primary]</div>
                    <small>{top_model['Name'][:30]}</small>
                </div>
                <div class="stat-box macro">
                    <div class="stat-number">{top_model.get('Macro F1', 0):.4f}</div>
                    <div class="stat-label">Macro F1</div>
                    <small>Unweighted class average</small>
                </div>
                <div class="stat-box accuracy">
                    <div class="stat-number">{top_model.get('Accuracy', 0):.4f}</div>
                    <div class="stat-label">Accuracy</div>
                    <small>Overall correctness</small>
                </div>
            </div>
        </div>
        
        <div class="container">
            <h2>Comprehensive Metrics Comparison</h2>
            <img src="../figures/comparison_plots/comprehensive_metrics.png" alt="Comprehensive Metrics">
        </div>
        
        <div class="container">
            <h2>Multi-Metric Radar Chart</h2>
            <img src="../figures/metrics_radar/metrics_radar.png" alt="Radar Chart">
        </div>
        
        <div class="container">
            <h2>Top 10 Models (Ranked by F1 Sum Score)</h2>
            {df_display.head(10).to_html(index=False)}
        </div>
        
        <div class="container">
            <h2>Class Definitions</h2>
            <ul>
                <li><strong>Class 0 (Lone-Lone):</strong> Both cells are noise - no particle cluster</li>
                <li><strong>Class 1 (True-True):</strong> Both cells belong to the same particle cluster</li>
                <li><strong>Class 2 (Cluster-Lone):</strong> Source cell in cluster, destination is noise</li>
                <li><strong>Class 3 (Lone-Cluster):</strong> Source is noise, destination in cluster</li>
                <li><strong>Class 4 (Cluster-Cluster):</strong> Cells belong to different clusters</li>
            </ul>
        </div>
        
        <div class="container" style="text-align: center; color: #7f8c8d;">
            <p><strong>Note:</strong> F1 Sum Score (FSS) is the recommended primary metric as it balances both precision and recall.</p>
            <p>Check individual model folders for ROC curves, PR curves, confusion matrices, and per-class performance breakdowns.</p>
        </div>
    </div>
</body>
</html>"""
    
    with open(os.path.join(folders['reports'], "master_report.html"), 'w') as f:
        f.write(html_report)
    print("   Done: master_report.html saved")


def main():
    global args  # Make args available to functions
    args = parse_args()
    
    print("="*100)
    print("CaloGraphNet - Model Analysis Suite")
    print("="*100)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n[PRIMARY METRIC] F1 Sum Score (FSS) - balances Precision AND Recall")
    print(f"   Models directory: {args.models_dir}")
    print(f"   Parquet directory: {args.parquet_dir}")
    print(f"   Output directory: {args.output_dir}")
    
    # Create output structure
    folders = create_output_structure(args.output_dir)
    
    # Find all models
    pkl_files = sorted(glob.glob(os.path.join(args.models_dir, "*_metrics.pkl")))
    model_names = [os.path.basename(f).replace("_metrics.pkl", "") for f in pkl_files]
    
    print(f"\nFound {len(model_names)} models:")
    for name in model_names:
        print(f"   - {name}")
    
    all_results = []
    
    print("\n" + "="*100)
    print("PHASE 1: Generating Per-Model Plots")
    print("="*100)
    
    for model_name in model_names:
        print(f"\nProcessing: {model_name}")
        try:
            # Load metrics
            pkl_path = os.path.join(args.models_dir, f"{model_name}_metrics.pkl")
            with open(pkl_path, 'rb') as f:
                metrics = pickle.load(f)
            
            # Extract metrics
            extracted = extract_model_metrics(metrics)
            
            # Get model configuration
            model_args = metrics.get('args', {})
            
            # Store results
            result = {
                'name': model_name,
                'architecture': model_args.get('model', 'unknown'),
                'loss': model_args.get('weight_strategy', 'cross_entropy') if model_args.get('weighted_loss', False) else 'cross_entropy',
                'features': 'baseline' if model_args.get('baseline', True) else 'all_features',
                **extracted
            }
            all_results.append(result)
            
            # Generate plots
            plot_roc_curves(model_name, extracted, folders, CLASS_NAMES, CLASS_COLORS)
            plot_pr_curves(model_name, extracted, folders, CLASS_NAMES, CLASS_COLORS, 
                          args.max_rows_roc, args.batch_size)
            plot_confusion_matrix_enhanced(model_name, extracted, folders, CLASS_NAMES, CLASS_COLORS,
                                          args.max_rows_confusion, args.batch_size)
            
            gc.collect()
            
        except Exception as e:
            print(f"      ERROR: {str(e)[:150]}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("\nNo models processed successfully!")
        return
    
    print("\n" + "="*100)
    print("PHASE 2: Creating Comparison Plots")
    print("="*100)
    
    plot_radar_chart(all_results, folders, args.top_n)
    plot_metric_comparison(all_results, folders, CLASS_NAMES)
    
    print("\n" + "="*100)
    print("PHASE 3: Generating Tables and Reports")
    print("="*100)
    
    # Create comprehensive dataframe
    df_summary = pd.DataFrame(all_results)
    df_summary = df_summary.sort_values('f1_sum_score', ascending=False)
    
    # Add star rating
    df_summary['Rating'] = pd.cut(df_summary['f1_sum_score'], 
                                   bins=[0, 1.5, 2.5, 3.5, 4.5, 5.0],
                                   labels=['1★', '2★', '3★', '4★', '5★'])
    
    # Select columns for display
    display_columns = ['name', 'architecture', 'loss', 'features', 'Rating',
                      'f1_sum_score', 'weighted_f1_score', 'randomness_metric',
                      'macro_f1', 'macro_recall', 'macro_precision', 'accuracy', 'best_epoch']
    
    df_display = df_summary[[c for c in display_columns if c in df_summary.columns]].copy()
    df_display.columns = [c.replace('_', ' ').title() for c in df_display.columns]
    
    # Save CSV
    df_summary.to_csv(os.path.join(folders['data'], "comprehensive_metrics.csv"), index=False)
    print("   Saved: comprehensive_metrics.csv")
    
    # Create styled HTML table
    styled = df_display.head(10).style\
        .background_gradient(subset=['F1 Sum Score', 'Weighted F1 Score', 'Macro F1'], cmap=GREEN_COLORMAP)\
        .background_gradient(subset=['Accuracy'], cmap=BLUE_COLORMAP)\
        .format({
            'F1 Sum Score': '{:.2f}',
            'Weighted F1 Score': '{:.3f}',
            'Randomness Metric': '{:.2f}',
            'Macro F1': '{:.4f}',
            'Macro Recall': '{:.4f}',
            'Macro Precision': '{:.4f}',
            'Accuracy': '{:.4f}'
        })\
        .set_caption('Model Performance Comparison (Ranked by F1 Sum Score)')
    
    styled.to_html(os.path.join(folders['tables'], "comprehensive_table.html"))
    print("   Saved: comprehensive_table.html")
    
    # Generate HTML report
    generate_html_report(all_results, df_display, folders, CLASS_NAMES, 
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)
    
    # Print rankings
    print("\nTOP 5 MODELS (by F1 Sum Score):")
    for i, (_, row) in enumerate(df_display.head(5).iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}th"
        print(f"   {medal} {row['Name'][:50]}: FSS={row['F1 Sum Score']:.2f}")
    
    print(f"\nAll outputs saved to: {args.output_dir}")
    print(f"Open report: {os.path.join(folders['reports'], 'master_report.html')}")


if __name__ == "__main__":
    main()
