"""
Gesture Recognition Evaluation Module
======================================

This module provides high-level evaluation functions for gesture recognizers
using the metrics utilities and data loaders.

Author: Expert Python Engineer & MLOps Specialist
Date: 2026-01-15
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import hashlib
import warnings

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Install with: pip install numpy")

# Import from refactored modules (use fully qualified package path for -m execution)
from src.recognizers.utils.metrics import GestureClassificationMetrics
from src.recognizers.data_utils.loaders import load_hagrid_samples


# Backward compatibility - re-export for existing imports
__all__ = ['GestureClassificationMetrics', 'load_hagrid_samples', 'evaluate_recognizer', 'demo_metrics_module']


def evaluate_recognizer(
    recognizer,
    test_data: List[Dict[str, Any]],
    class_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluates the GestureRecognizer against a labeled test set using comprehensive metrics.
    
    Args:
        recognizer: Your GestureRecognizer instance
        test_data: A list of dicts with 'landmarks', 'fingers', 'dist', and 'label'
        class_names: Optional list of gesture class names
        output_dir: Optional output directory for saving results
        
    Returns:
        Dictionary containing all evaluation metrics
    
    Note:
        'unknown' predictions are included in evaluation as they represent recognition failures,
        which is important for understanding the true performance of the recognizer.
    """
    # Initialize output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'docs/evaluation_result/recognizer_evaluator'
    
    # Collect predictions
    y_true = []
    y_pred = []
    unknown_count = 0
    
    print(f"--- Starting Evaluation on {len(test_data)} samples ---")
    
    for sample in test_data:
        # Get prediction from recognizer
        prediction = recognizer.recognize_gesture(
            fingers=sample['fingers'],
            landmark_list=sample['landmarks'],
            distance_thumb_index=sample.get('dist')
        )
        
        # Track unknown predictions for reporting
        if prediction == 'unknown':
            unknown_count += 1
        
        y_true.append(sample['label'])
        y_pred.append(prediction)
    
    if unknown_count > 0:
        print(f"Note: {unknown_count} samples predicted as 'unknown' ({unknown_count/len(test_data)*100:.1f}%)")
    
    print(f"Evaluating on {len(y_true)} samples (including 'unknown' predictions)")
    
    # Convert labels to indices if they are strings
    if len(y_true) > 0 and isinstance(y_true[0], str):
        unique_labels = sorted(list(set(y_true + y_pred)))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y_true_idx = [label_to_idx[label] for label in y_true]
        y_pred_idx = [label_to_idx[label] for label in y_pred]
        
        # Prepare class names for all unique labels found
        if class_names is None:
            class_names = unique_labels
        else:
            # Ensure all unique labels are in class_names list
            # Add 'unknown' if it appears in predictions
            if 'unknown' in unique_labels and 'unknown' not in class_names:
                class_names = list(class_names) + ['unknown']
            # Filter to only labels present in evaluation
            class_names = [name for name in class_names if name in unique_labels]
            # Ensure ordering matches unique_labels
            class_names = sorted(class_names, key=lambda x: unique_labels.index(x))
    else:
        y_true_idx = y_true
        y_pred_idx = y_pred
        if class_names is None:
            class_names = [f"Class_{i}" for i in sorted(set(y_true + y_pred))]
    
    # Initialize metrics calculator with all class names
    metrics_calc = GestureClassificationMetrics(
        class_names=class_names,
        output_dir=output_dir
    )
    
    # Compute all metrics
    metrics = metrics_calc.compute_all_metrics(
        y_true=y_true_idx,
        y_pred=y_pred_idx,
        top_k=3
    )
    
    # Add statistics to metadata
    metrics['unknown_prediction_count'] = unknown_count
    metrics['unknown_prediction_rate'] = unknown_count / len(test_data) if len(test_data) > 0 else 0.0
    metrics['total_samples'] = len(test_data)
    metrics['evaluated_samples'] = len(y_true)
    
    # Print summary
    metrics_calc.print_summary(metrics, detailed=True)
    
    # Print additional statistics about unknown predictions
    if unknown_count > 0:
        print("\n" + "="*70)
        print(" RECOGNITION FAILURE ANALYSIS ".center(70, "="))
        print("="*70)
        print(f"\nUnknown Predictions: {unknown_count} / {len(test_data)} ({unknown_count/len(test_data)*100:.1f}%)")
        print("\nNote: 'unknown' represents cases where the recognizer could not")
        print("confidently classify the gesture. This is included in the evaluation")
        print("to reflect the true end-to-end performance of the system.")
        print("="*70 + "\n")
    
    # Generate a unique run_id for this evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_hash = hashlib.md5(str(metrics.get('accuracy', 0)).encode()).hexdigest()[:8]
    run_id = f"{timestamp}_{metrics_hash}"
    
    # Save metrics to file with versioning
    metrics_calc.save_metrics_to_file(metrics, format='txt', run_id=run_id)
    metrics_calc.save_metrics_to_file(metrics, format='json', run_id=run_id)
    metrics_calc.save_metrics_to_file(metrics, format='csv', run_id=run_id)
    
    # Generate visualizations with run_id
    metrics_calc.plot_confusion_matrix(y_true_idx, y_pred_idx, normalize=None, run_id=run_id)
    metrics_calc.plot_confusion_matrix(
        y_true_idx, y_pred_idx, 
        normalize='true',
        title='Confusion Matrix (Normalized by True Label)',
        run_id=run_id
    )
    metrics_calc.plot_per_class_metrics(y_true_idx, y_pred_idx, run_id=run_id)
    
    print(f"\n✓ Evaluation complete! Run ID: {run_id}")
    print(f"✓ Results saved to: {output_dir}")
    
    return metrics


# ============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# ============================================================================

def demo_metrics_module():
    """
    Demonstrate the usage of GestureClassificationMetrics with synthetic data.
    """
    print("\n" + "="*70)
    print(" DEMO: Comprehensive Gesture Classification Metrics ".center(70, "="))
    print("="*70 + "\n")
    
    # Synthetic example data
    np.random.seed(42)
    n_samples = 200
    n_classes = 6
    
    # Simulate predictions with some class imbalance
    y_true = np.random.choice(n_classes, size=n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.05])
    
    # Simulate predictions with ~85% accuracy
    y_pred = y_true.copy()
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    y_pred[noise_idx] = np.random.choice(n_classes, size=len(noise_idx))
    
    # Simulate probability predictions for top-K accuracy
    y_pred_proba = np.random.rand(n_samples, n_classes)
    for i in range(n_samples):
        y_pred_proba[i, y_pred[i]] += 2.0  # Make true prediction more likely
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)  # Normalize
    
    # Define class names
    class_names = ['fist', 'palm', 'like', 'peace', 'ok', 'stop']
    
    # Initialize metrics calculator
    metrics_calc = GestureClassificationMetrics(
        class_names=class_names,
        output_dir='./demo_output'
    )
    
    print("Computing all metrics...")
    
    # Compute all metrics
    metrics = metrics_calc.compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        top_k=3
    )
    
    # Print summary
    metrics_calc.print_summary(metrics, detailed=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    metrics_calc.plot_confusion_matrix(y_true, y_pred)
    metrics_calc.plot_confusion_matrix(y_true, y_pred, normalize='true',
                                      save_path='./demo_output/confusion_matrix_normalized.png')
    metrics_calc.plot_per_class_metrics(y_true, y_pred)
    
    # Save metrics
    print("\nSaving metrics to files...")
    metrics_calc.save_metrics_to_file(metrics, format='txt')
    metrics_calc.save_metrics_to_file(metrics, format='json')
    metrics_calc.save_metrics_to_file(metrics, format='csv')
    
    print("\n" + "="*70)
    print(" Demo completed! Check './demo_output' folder for results. ".center(70, "="))
    print("="*70 + "\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        # Run demo with synthetic data
        demo_metrics_module()
    else:
        # Run full evaluation with HAGRID dataset
        from src.recognizers.models.gesture_recognizer import GestureRecognizer
        
        data_dir = '/home/toantim/ToanFolder/Hand-gesture-controlled/data/hagrid-sample-30k-384p'
        labels = ['fist', 'palm', 'like', 'one', 'peace', 'ok']
        
        print("Loading HAGRID dataset samples...")
        test_set = load_hagrid_samples(data_dir, labels, samples_per_label=50)
        
        print(f"\nTotal samples loaded: {len(test_set)}")
        
        # Evaluate recognizer with comprehensive metrics
        gr = GestureRecognizer()
        evaluate_recognizer(gr, test_set, class_names=labels)