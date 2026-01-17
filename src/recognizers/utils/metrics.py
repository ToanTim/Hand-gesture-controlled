"""
Comprehensive Evaluation Metrics Module for Multi-Class Classification
========================================================================

This module provides all standard evaluation metrics for gesture recognition
and other multi-class classification tasks. It handles class imbalance,
provides visualizations, and is designed for MLOps integration.

Author: Expert Python Engineer & MLOps Specialist
Date: 2026-01-15
"""

from typing import Union, List, Dict, Optional, Tuple, Any
from pathlib import Path
import warnings
from datetime import datetime
import hashlib
import json as json_module

# Core dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Install with: pip install numpy")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available. Install with: pip install pandas")

# Core ML metrics
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
        top_k_accuracy_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Install with: pip install scikit-learn")

# Visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    warnings.warn("Seaborn not available. Install with: pip install seaborn")

# Optional deep learning framework support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class GestureClassificationMetrics:
    """
    Comprehensive metrics calculator for multi-class gesture recognition.
    
    Features:
    - All standard classification metrics (accuracy, precision, recall, F1)
    - Confusion matrix with visualization
    - Top-K accuracy
    - Class imbalance handling
    - Support for numpy, PyTorch, and TensorFlow inputs
    - Modular design for training loops and evaluation scripts
    """
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Initialize the metrics calculator.
        
        Args:
            class_names: List of class names in order. If None, uses indices.
            output_dir: Directory to save visualizations. If None, uses current dir.
            figsize: Default figure size for visualizations.
        """
        self.class_names = class_names
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        self.metrics_cache = {}
    
    def _convert_to_numpy(
        self,
        data: Union['np.ndarray', List, 'torch.Tensor', 'tf.Tensor']
    ) -> 'np.ndarray':
        """
        Convert input data to numpy array from various formats.
        
        Args:
            data: Input data (numpy, list, PyTorch tensor, or TensorFlow tensor)
            
        Returns:
            numpy.ndarray
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required. Install with: pip install numpy")
            
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, list):
            return np.array(data)
        elif TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        elif TF_AVAILABLE and isinstance(data, (tf.Tensor, tf.Variable)):
            return data.numpy()
        else:
            try:
                return np.array(data)
            except Exception as e:
                raise TypeError(
                    f"Cannot convert type {type(data)} to numpy array. "
                    f"Supported types: numpy.ndarray, list, torch.Tensor, tf.Tensor"
                ) from e
    
    def compute_all_metrics(
        self,
        y_true: Union['np.ndarray', List, 'torch.Tensor', 'tf.Tensor'],
        y_pred: Union['np.ndarray', List, 'torch.Tensor', 'tf.Tensor'],
        y_pred_proba: Optional[Union['np.ndarray', 'torch.Tensor', 'tf.Tensor']] = None,
        top_k: int = 3,
        average_methods: List[str] = ['macro', 'weighted'],
        zero_division: int = 0
    ) -> Dict[str, Any]:
        """
        Compute all classification metrics at once.
        
        Args:
            y_true: Ground truth labels (shape: [n_samples])
            y_pred: Predicted labels (shape: [n_samples])
            y_pred_proba: Prediction probabilities (shape: [n_samples, n_classes]).
                         Required for top-K accuracy.
            top_k: K value for top-K accuracy
            average_methods: List of averaging methods for precision/recall/F1
            zero_division: Value to return when there is a zero division
            
        Returns:
            Dictionary containing all metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required. Install with: pip install numpy")
            
        # Convert inputs to numpy
        y_true = self._convert_to_numpy(y_true)
        y_pred = self._convert_to_numpy(y_pred)
        if y_pred_proba is not None:
            y_pred_proba = self._convert_to_numpy(y_pred_proba)
        
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true has {len(y_true)} samples, "
                f"y_pred has {len(y_pred)} samples"
            )
        
        metrics = {}
        
        # 1. Overall Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 2. Per-class and averaged precision, recall, F1-score
        for avg_method in average_methods:
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred,
                average=avg_method,
                zero_division=zero_division
            )
            metrics[f'precision_{avg_method}'] = precision
            metrics[f'recall_{avg_method}'] = recall
            metrics[f'f1_{avg_method}'] = f1
        
        # 3. Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(
                y_true, y_pred,
                average=None,
                zero_division=zero_division
            )
        
        metrics['precision_per_class'] = precision_per_class
        metrics['recall_per_class'] = recall_per_class
        metrics['f1_per_class'] = f1_per_class
        metrics['support_per_class'] = support_per_class
        metrics['support_total'] = int(support_per_class.sum())
        
        # 4. Confusion Matrix
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        metrics['confusion_matrix'] = cm
        metrics['unique_labels'] = unique_labels
        
        # 5. Top-K Accuracy (if probabilities provided)
        if y_pred_proba is not None:
            try:
                top_k_acc = top_k_accuracy_score(
                    y_true, y_pred_proba,
                    k=top_k,
                    labels=unique_labels
                )
                metrics[f'top_{top_k}_accuracy'] = top_k_acc
            except Exception as e:
                warnings.warn(f"Could not compute top-{top_k} accuracy: {e}")
                metrics[f'top_{top_k}_accuracy'] = None
        
        # 6. Class imbalance info
        unique, counts = np.unique(y_true, return_counts=True)
        metrics['class_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
        metrics['imbalance_ratio'] = float(counts.max()) / float(counts.min()) if len(counts) > 1 else 1.0
        
        # Cache for later use
        self.metrics_cache = metrics
        
        return metrics
    
    def compute_accuracy(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List]
    ) -> float:
        """
        Compute overall accuracy.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score
        """
        y_true = self._convert_to_numpy(y_true)
        y_pred = self._convert_to_numpy(y_pred)
        return accuracy_score(y_true, y_pred)
    
    def compute_precision_recall_f1(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        average: str = 'macro',
        zero_division: int = 0
    ) -> Tuple[float, float, float, int]:
        """
        Compute precision, recall, and F1-score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            average: Averaging method ('macro', 'micro', 'weighted', None)
            zero_division: Value to return when there is a zero division
            
        Returns:
            Tuple of (precision, recall, f1, support)
        """
        y_true = self._convert_to_numpy(y_true)
        y_pred = self._convert_to_numpy(y_pred)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred,
            average=average,
            zero_division=zero_division
        )
        
        return precision, recall, f1, support
    
    def compute_confusion_matrix(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            normalize: Normalization mode ('true', 'pred', 'all', or None)
            
        Returns:
            Confusion matrix as numpy array
        """
        y_true = self._convert_to_numpy(y_true)
        y_pred = self._convert_to_numpy(y_pred)
        
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        return cm
    
    def compute_top_k_accuracy(
        self,
        y_true: Union[np.ndarray, List],
        y_pred_proba: Union[np.ndarray, 'torch.Tensor', 'tf.Tensor'],
        k: int = 3
    ) -> float:
        """
        Compute top-K accuracy.
        
        Useful when the UI can suggest multiple gestures and we want to know
        if the correct gesture is in the top K predictions.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Prediction probabilities (shape: [n_samples, n_classes])
            k: Number of top predictions to consider
            
        Returns:
            Top-K accuracy score
        """
        y_true = self._convert_to_numpy(y_true)
        y_pred_proba = self._convert_to_numpy(y_pred_proba)
        
        return top_k_accuracy_score(y_true, y_pred_proba, k=k)
    
    def get_classification_report(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        output_dict: bool = False,
        zero_division: int = 0
    ) -> Union[str, Dict]:
        """
        Generate scikit-learn classification report.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            output_dict: If True, return as dictionary; else as string
            zero_division: Value to return when there is a zero division
            
        Returns:
            Classification report (string or dict)
        """
        y_true = self._convert_to_numpy(y_true)
        y_pred = self._convert_to_numpy(y_pred)
        
        target_names = None
        if self.class_names is not None:
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            target_names = [self.class_names[i] if i < len(self.class_names) else f"Class_{i}" 
                           for i in unique_labels]
        
        return classification_report(
            y_true, y_pred,
            target_names=target_names,
            output_dict=output_dict,
            zero_division=zero_division
        )
    
    def plot_confusion_matrix(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        normalize: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = False,
        cmap: str = 'Blues',
        title: str = 'Confusion Matrix',
        create_timestamped_folder: bool = True,
        run_id: Optional[str] = None
    ) -> 'plt.Figure':
        """
        Create and optionally save confusion matrix heatmap.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            normalize: Normalization mode ('true', 'pred', 'all', or None)
            save_path: Path to save the figure. If None, saves to output_dir
            show_plot: Whether to display the plot
            cmap: Colormap for the heatmap
            title: Plot title
            create_timestamped_folder: If True, creates folder with format MM_HH_DD_MM_YYYY
            run_id: Unique identifier for this run (used in filename)
            
        Returns:
            matplotlib Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for visualization. Install with: pip install matplotlib")
        if not SEABORN_AVAILABLE:
            raise ImportError("Seaborn is required for visualization. Install with: pip install seaborn")
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
            
        y_true = self._convert_to_numpy(y_true)
        y_pred = self._convert_to_numpy(y_pred)
        
        # Compute confusion matrix
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels, normalize=normalize)
        
        # Prepare labels
        if self.class_names is not None:
            labels = [self.class_names[i] if i < len(self.class_names) else f"Class_{i}" 
                     for i in unique_labels]
        else:
            labels = [f"Class_{i}" for i in unique_labels]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(
            cm, annot=True,
            fmt='.2f' if normalize else 'd',
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Determine save path
        if save_path is None:
            if create_timestamped_folder:
                now = datetime.now()
                timestamped_folder = now.strftime("%M_%H_%d_%m_%Y")
                save_dir = self.output_dir / timestamped_folder
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = self.output_dir
            
            # Generate filename with run_id if provided
            if run_id:
                filename = f'confusion_matrix_{run_id}.png'
            else:
                norm_suffix = f'_{normalize}' if normalize else ''
                filename = f'confusion_matrix{norm_suffix}.png'
            
            save_path = save_dir / filename
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def plot_per_class_metrics(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        save_path: Optional[Union[str, Path]] = None,
        show_plot: bool = False,
        metrics_to_plot: List[str] = ['precision', 'recall', 'f1'],
        create_timestamped_folder: bool = True,
        run_id: Optional[str] = None
    ) -> plt.Figure:
        """
        Create bar plot of per-class metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            metrics_to_plot: List of metrics to plot
            create_timestamped_folder: If True, creates folder with format MM_HH_DD_MM_YYYY
            run_id: Unique identifier for this run (used in filename)
            
        Returns:
            matplotlib Figure object
        """
        y_true = self._convert_to_numpy(y_true)
        y_pred = self._convert_to_numpy(y_pred)
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas is required for visualization. Install with: pip install pandas")
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for visualization. Install with: pip install matplotlib")
        
        # Compute per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred,
            average=None,
            zero_division=0
        )
        
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        
        # Prepare labels
        if self.class_names is not None:
            labels = [self.class_names[i] if i < len(self.class_names) else f"Class_{i}" 
                     for i in unique_labels]
        else:
            labels = [f"Class_{i}" for i in unique_labels]
        
        # Create DataFrame for easy plotting
        metrics_dict = {}
        if 'precision' in metrics_to_plot:
            metrics_dict['Precision'] = precision
        if 'recall' in metrics_to_plot:
            metrics_dict['Recall'] = recall
        if 'f1' in metrics_to_plot:
            metrics_dict['F1-Score'] = f1
        
        df = pd.DataFrame(metrics_dict, index=labels)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_xlabel('Gesture Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Determine save path
        if save_path is None:
            if create_timestamped_folder:
                now = datetime.now()
                timestamped_folder = now.strftime("%M_%H_%d_%m_%Y")
                save_dir = self.output_dir / timestamped_folder
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                save_dir = self.output_dir
            
            # Generate filename with run_id if provided
            if run_id:
                filename = f'per_class_metrics_{run_id}.png'
            else:
                filename = 'per_class_metrics.png'
            
            save_path = save_dir / filename
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def print_summary(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        detailed: bool = True
    ) -> None:
        """
        Print a formatted summary of all metrics.
        
        Args:
            metrics: Metrics dictionary (uses cached if None)
            detailed: Whether to print detailed per-class metrics
        """
        if metrics is None:
            metrics = self.metrics_cache
        
        if not metrics:
            print("No metrics available. Run compute_all_metrics() first.")
            return
        
        print("\n" + "="*70)
        print(" GESTURE RECOGNITION EVALUATION SUMMARY ".center(70, "="))
        print("="*70 + "\n")
        
        # Overall metrics
        print(f"{'OVERALL METRICS':<30}")
        print("-" * 70)
        print(f"{'Accuracy:':<30} {metrics.get('accuracy', 0):.4f}")
        
        if 'precision_macro' in metrics:
            print(f"{'Precision (Macro Avg):':<30} {metrics['precision_macro']:.4f}")
        if 'recall_macro' in metrics:
            print(f"{'Recall (Macro Avg):':<30} {metrics['recall_macro']:.4f}")
        if 'f1_macro' in metrics:
            print(f"{'F1-Score (Macro Avg):':<30} {metrics['f1_macro']:.4f}")
        
        print()
        
        if 'precision_weighted' in metrics:
            print(f"{'Precision (Weighted Avg):':<30} {metrics['precision_weighted']:.4f}")
        if 'recall_weighted' in metrics:
            print(f"{'Recall (Weighted Avg):':<30} {metrics['recall_weighted']:.4f}")
        if 'f1_weighted' in metrics:
            print(f"{'F1-Score (Weighted Avg):':<30} {metrics['f1_weighted']:.4f}")
        
        # Top-K accuracy
        top_k_key = [k for k in metrics.keys() if k.startswith('top_') and k.endswith('_accuracy')]
        if top_k_key:
            for key in top_k_key:
                k_val = key.split('_')[1]
                acc_val = metrics[key]
                if acc_val is not None:
                    print(f"{'Top-' + k_val + ' Accuracy:':<30} {acc_val:.4f}")
        
        print()
        
        # Class imbalance info
        if 'imbalance_ratio' in metrics:
            print(f"{'CLASS IMBALANCE INFO':<30}")
            print("-" * 70)
            print(f"{'Imbalance Ratio:':<30} {metrics['imbalance_ratio']:.2f}")
            if 'class_distribution' in metrics:
                print(f"{'Class Distribution:':<30}")
                for cls, count in metrics['class_distribution'].items():
                    cls_name = self.class_names[cls] if self.class_names and cls < len(self.class_names) else f"Class_{cls}"
                    print(f"  {cls_name:<28} {count} samples")
        
        # Per-class metrics (detailed)
        if detailed and 'precision_per_class' in metrics:
            print("\n" + f"{'PER-CLASS METRICS':<30}")
            print("-" * 70)
            print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
            print("-" * 70)
            
            unique_labels = metrics.get('unique_labels', [])
            for i, label in enumerate(unique_labels):
                cls_name = self.class_names[label] if self.class_names and label < len(self.class_names) else f"Class_{label}"
                precision = metrics['precision_per_class'][i]
                recall = metrics['recall_per_class'][i]
                f1 = metrics['f1_per_class'][i]
                support = int(metrics['support_per_class'][i])
                
                print(f"{cls_name:<20} {precision:>11.4f} {recall:>11.4f} {f1:>11.4f} {support:>9}")
        
        print("\n" + "="*70 + "\n")
    
    def save_metrics_to_file(
        self,
        metrics: Optional[Dict[str, Any]] = None,
        filepath: Optional[Union[str, Path]] = None,
        format: str = 'txt',
        run_id: Optional[str] = None,
        version: Optional[str] = None,
        create_timestamped_folder: bool = True
    ) -> str:
        """
        Save metrics to a file with version tracking and unique ID.
        
        Args:
            metrics: Metrics dictionary (uses cached if None)
            filepath: Path to save file. If None, uses output_dir
            format: File format ('txt', 'json', 'csv')
            run_id: Unique identifier for this evaluation run. If None, generates auto-ID
            version: Version string (e.g., 'v1.0', 'model_v2'). If None, uses timestamp
            create_timestamped_folder: If True, creates folder with format MM_HH_DD_MM_YYYY
            
        Returns:
            Path to saved file
        """
        if metrics is None:
            metrics = self.metrics_cache
        
        if not metrics:
            print("No metrics available to save.")
            return ""
        
        # Generate run_id if not provided
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_hash = hashlib.md5(str(metrics.get('accuracy', 0)).encode()).hexdigest()[:8]
            run_id = f"{timestamp}_{metrics_hash}"
        
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create timestamped folder if requested
        if create_timestamped_folder:
            now = datetime.now()
            timestamped_folder = now.strftime("%M_%H_%d_%m_%Y")
            save_dir = self.output_dir / timestamped_folder
        else:
            save_dir = self.output_dir
        
        # Ensure parent directory exists
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if filepath is None:
            filepath = save_dir / f'metrics_report_{run_id}.{format}'
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        metadata = {
            'run_id': run_id,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'format': format,
            'timestamped_folder': timestamped_folder if create_timestamped_folder else None
        }
        
        if format == 'txt':
            with open(filepath, 'w') as f:
                f.write("="*70 + "\n")
                f.write(" GESTURE RECOGNITION EVALUATION REPORT ".center(70, "=") + "\n")
                f.write("="*70 + "\n\n")
                
                # Write metadata
                f.write("METADATA\n")
                f.write("-" * 70 + "\n")
                f.write(f"Run ID:       {metadata['run_id']}\n")
                f.write(f"Version:      {metadata['version']}\n")
                f.write(f"Timestamp:    {metadata['timestamp']}\n")
                if metadata['timestamped_folder']:
                    f.write(f"Folder:       {metadata['timestamped_folder']}\n")
                
                # Add sample statistics
                if 'total_samples' in metrics:
                    f.write(f"Total Samples: {metrics['total_samples']}\n")
                if 'evaluated_samples' in metrics:
                    f.write(f"Evaluated:     {metrics['evaluated_samples']}\n")
                if 'unknown_prediction_count' in metrics and metrics['unknown_prediction_count'] > 0:
                    unknown_count = metrics['unknown_prediction_count']
                    unknown_rate = metrics.get('unknown_prediction_rate', 0.0)
                    f.write(f"Unknown Preds: {unknown_count} ({unknown_rate*100:.1f}%)\n")
                f.write("\n")
                
                # Write overall metrics
                f.write("OVERALL METRICS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Overall Accuracy: {metrics.get('accuracy', 0):.4f}\n\n")
                
                if 'precision_macro' in metrics:
                    f.write(f"Macro Averaged Metrics:\n")
                    f.write(f"  Precision: {metrics['precision_macro']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall_macro']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1_macro']:.4f}\n\n")
                
                if 'precision_weighted' in metrics:
                    f.write(f"Weighted Averaged Metrics:\n")
                    f.write(f"  Precision: {metrics['precision_weighted']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall_weighted']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1_weighted']:.4f}\n\n")
                
                # Class distribution
                if 'class_distribution' in metrics:
                    f.write(f"Class Distribution:\n")
                    for cls, count in metrics['class_distribution'].items():
                        cls_name = self.class_names[cls] if self.class_names and cls < len(self.class_names) else f"Class_{cls}"
                        f.write(f"  {cls_name}: {count} samples\n")
            
            print(f"Metrics report saved to: {filepath}")
        
        elif format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value
            
            # Add metadata
            output_data = {
                'metadata': metadata,
                'metrics': serializable_metrics
            }
            
            with open(filepath, 'w') as f:
                json_module.dump(output_data, f, indent=2)
            
            print(f"Metrics report saved to: {filepath}")
        
        elif format == 'csv':
            # Create DataFrame for per-class metrics
            if 'precision_per_class' in metrics:
                unique_labels = metrics.get('unique_labels', [])
                data = {
                    'run_id': [run_id] * len(unique_labels),
                    'version': [version] * len(unique_labels),
                    'class': [self.class_names[l] if self.class_names and l < len(self.class_names) else f"Class_{l}" 
                             for l in unique_labels],
                    'precision': metrics['precision_per_class'],
                    'recall': metrics['recall_per_class'],
                    'f1_score': metrics['f1_per_class'],
                    'support': metrics['support_per_class']
                }
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False)
                print(f"Per-class metrics saved to: {filepath}")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'txt', 'json', or 'csv'.")
        
        return str(filepath)
