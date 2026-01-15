# Comprehensive Evaluation Metrics Module - Usage Guide

## Overview

This module provides **production-ready evaluation metrics** for multi-class classification tasks, specifically designed for gesture recognition with class imbalance and visually similar gestures in mind.

## Features

✅ **All Standard Metrics**

- Overall Accuracy
- Precision, Recall, F1-Score (per-class, macro, weighted)
- Confusion Matrix (raw counts and normalized)
- Top-K Accuracy (configurable K)

✅ **Advanced Capabilities**

- Handles class imbalance gracefully
- Support for numpy, PyTorch, and TensorFlow tensors
- Beautiful visualizations (confusion matrix heatmaps, per-class bar charts)
- Multiple output formats (TXT, JSON, CSV)
- Modular design for training loops and evaluation scripts

✅ **MLOps Ready**

- Clean API for integration
- Comprehensive logging
- Reproducible results
- Export-friendly formats

---

## Installation

The module uses common ML libraries:

```bash
pip install numpy scikit-learn matplotlib seaborn pandas

# Optional: for deep learning framework support
pip install torch  # PyTorch
pip install tensorflow  # TensorFlow
```

---

## Quick Start

### 1. Basic Usage with Numpy Arrays

```python
from eva_recog_metrics import GestureClassificationMetrics
import numpy as np

# Your ground truth and predictions
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1])

# Define class names (optional but recommended)
class_names = ['fist', 'palm', 'peace']

# Initialize metrics calculator
metrics_calc = GestureClassificationMetrics(
    class_names=class_names,
    output_dir='./evaluation_results'
)

# Compute all metrics at once
metrics = metrics_calc.compute_all_metrics(
    y_true=y_true,
    y_pred=y_pred
)

# Print beautiful summary
metrics_calc.print_summary(metrics, detailed=True)
```

**Output:**

```
======================================================================
================ GESTURE RECOGNITION EVALUATION SUMMARY ==============
======================================================================

OVERALL METRICS
----------------------------------------------------------------------
Accuracy:                      0.7778
Precision (Macro Avg):         0.7778
Recall (Macro Avg):            0.7778
F1-Score (Macro Avg):          0.7778
...
```

---

### 2. Using with PyTorch

```python
import torch
from eva_recog_metrics import GestureClassificationMetrics

# PyTorch tensors from your model
y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = torch.tensor([0, 1, 2, 0, 2, 2, 0, 1, 1])

# The module automatically converts tensors to numpy
metrics_calc = GestureClassificationMetrics(
    class_names=['fist', 'palm', 'peace']
)

metrics = metrics_calc.compute_all_metrics(y_true, y_pred)
```

---

### 3. Top-K Accuracy (Multi-Gesture Suggestions)

If your UI can suggest multiple gestures, use Top-K accuracy:

```python
# Prediction probabilities (shape: [n_samples, n_classes])
y_pred_proba = np.array([
    [0.7, 0.2, 0.1],  # Most confident: class 0
    [0.1, 0.8, 0.1],  # Most confident: class 1
    [0.3, 0.3, 0.4],  # Most confident: class 2
    # ... more samples
])

metrics = metrics_calc.compute_all_metrics(
    y_true=y_true,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    top_k=3  # Check if correct class is in top 3 predictions
)

print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
```

---

### 4. Visualizations

#### Confusion Matrix Heatmap

```python
# Standard confusion matrix
metrics_calc.plot_confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    normalize=None,  # Raw counts
    save_path='./confusion_matrix.png',
    show_plot=False
)

# Normalized by true labels (shows recall per class)
metrics_calc.plot_confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    normalize='true',  # Normalize by row
    save_path='./confusion_matrix_normalized.png'
)
```

#### Per-Class Metrics Bar Chart

```python
metrics_calc.plot_per_class_metrics(
    y_true=y_true,
    y_pred=y_pred,
    metrics_to_plot=['precision', 'recall', 'f1'],
    save_path='./per_class_metrics.png'
)
```

---

### 5. Exporting Metrics

#### Save to Text File

```python
metrics_calc.save_metrics_to_file(
    metrics=metrics,
    filepath='./metrics_report.txt',
    format='txt'
)
```

#### Save to JSON (for logging/MLOps)

```python
metrics_calc.save_metrics_to_file(
    metrics=metrics,
    filepath='./metrics.json',
    format='json'
)
```

#### Save to CSV (for spreadsheets)

```python
metrics_calc.save_metrics_to_file(
    metrics=metrics,
    filepath='./per_class_metrics.csv',
    format='csv'
)
```

---

## Advanced Usage

### Integration in Training Loop

```python
from eva_recog_metrics import GestureClassificationMetrics

class GestureTrainer:
    def __init__(self, model, class_names):
        self.model = model
        self.metrics_calc = GestureClassificationMetrics(
            class_names=class_names,
            output_dir='./training_logs/metrics'
        )

    def evaluate_epoch(self, dataloader, epoch):
        """Evaluate model after each epoch"""
        y_true_all = []
        y_pred_all = []
        y_pred_proba_all = []

        for batch in dataloader:
            inputs, labels = batch
            outputs = self.model(inputs)
            predictions = torch.argmax(outputs, dim=1)

            y_true_all.append(labels.cpu())
            y_pred_all.append(predictions.cpu())
            y_pred_proba_all.append(torch.softmax(outputs, dim=1).cpu())

        # Concatenate all batches
        y_true = torch.cat(y_true_all)
        y_pred = torch.cat(y_pred_all)
        y_pred_proba = torch.cat(y_pred_proba_all)

        # Compute metrics
        metrics = self.metrics_calc.compute_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            top_k=3
        )

        # Log to MLOps platform (Weights & Biases, MLflow, etc.)
        wandb.log({
            'epoch': epoch,
            'val/accuracy': metrics['accuracy'],
            'val/f1_macro': metrics['f1_macro'],
            'val/top_3_accuracy': metrics['top_3_accuracy']
        })

        # Save visualizations every 10 epochs
        if epoch % 10 == 0:
            self.metrics_calc.plot_confusion_matrix(
                y_true, y_pred,
                save_path=f'./logs/confusion_matrix_epoch_{epoch}.png'
            )

        return metrics
```

---

### Handling Class Imbalance

The module automatically computes and reports class imbalance:

```python
metrics = metrics_calc.compute_all_metrics(y_true, y_pred)

# Check imbalance ratio
imbalance_ratio = metrics['imbalance_ratio']
if imbalance_ratio > 5:
    print(f"⚠️ High class imbalance detected (ratio: {imbalance_ratio:.2f})")
    print("Consider using weighted metrics or class balancing techniques.")

# View class distribution
print("\nClass Distribution:")
for cls, count in metrics['class_distribution'].items():
    print(f"  {class_names[cls]}: {count} samples")
```

**Tip:** Use `weighted` averages when dealing with class imbalance:

- `precision_weighted`, `recall_weighted`, `f1_weighted`

---

### Individual Metric Functions

You can also compute metrics individually:

```python
# Just accuracy
accuracy = metrics_calc.compute_accuracy(y_true, y_pred)

# Just precision, recall, F1
precision, recall, f1, support = metrics_calc.compute_precision_recall_f1(
    y_true, y_pred,
    average='macro'
)

# Just confusion matrix
cm = metrics_calc.compute_confusion_matrix(
    y_true, y_pred,
    normalize='true'  # or None, 'pred', 'all'
)

# Just top-K accuracy
top_3_acc = metrics_calc.compute_top_k_accuracy(
    y_true, y_pred_proba,
    k=3
)

# Classification report (scikit-learn style)
report = metrics_calc.get_classification_report(
    y_true, y_pred,
    output_dict=False  # Set True for dict format
)
print(report)
```

---

## Integration with Existing Code

### Example: Evaluating Your Gesture Recognizer

```python
from eva_recog_metrics import evaluate_recognizer, load_hagrid_samples
from recognizers.gesture_recognizer import GestureRecognizer

# Load test data
data_dir = './data/hagrid-sample-30k-384p'
labels = ['fist', 'palm', 'like', 'peace', 'ok', 'stop']
test_data = load_hagrid_samples(data_dir, labels, samples_per_label=50)

# Initialize your recognizer
recognizer = GestureRecognizer()

# Evaluate with comprehensive metrics
metrics = evaluate_recognizer(
    recognizer=recognizer,
    test_data=test_data,
    class_names=labels,
    output_dir='./evaluation_results'
)

# All metrics, visualizations, and reports are automatically generated!
```

---

## Running the Demo

To see the module in action with synthetic data:

```bash
# From the evaluation directory
cd src/recognizers/evaluation/

# Run demo
python eva_recog_metrics.py --demo
```

This will:

1. Generate synthetic gesture classification data
2. Compute all metrics
3. Create visualizations
4. Save reports in multiple formats
5. Output everything to `./demo_output/`

---

## Output Files

When you run the evaluation, the following files are generated:

```
evaluation_results/
├── metrics_report.txt          # Human-readable report
├── metrics_report.json         # Machine-readable metrics
├── per_class_metrics.csv       # Spreadsheet-friendly format
├── confusion_matrix.png        # Heatmap (raw counts)
├── confusion_matrix_normalized.png  # Heatmap (normalized)
└── per_class_metrics.png       # Bar chart of precision/recall/F1
```

---

## Best Practices

### 1. **Always Use Class Names**

```python
# Good
metrics_calc = GestureClassificationMetrics(
    class_names=['fist', 'palm', 'peace', 'ok', 'stop']
)

# Avoid (will use "Class_0", "Class_1", etc.)
metrics_calc = GestureClassificationMetrics()
```

### 2. **Use Weighted Averages for Imbalanced Data**

```python
metrics = metrics_calc.compute_all_metrics(
    y_true, y_pred,
    average_methods=['macro', 'weighted']  # Include both
)

# Report weighted metrics when class sizes vary significantly
print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
```

### 3. **Leverage Top-K for Multi-Suggestion UI**

```python
# If your app shows top 3 gesture suggestions
metrics = metrics_calc.compute_all_metrics(
    y_true, y_pred, y_pred_proba,
    top_k=3
)

# User satisfaction metric: is correct gesture in top 3?
print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
```

### 4. **Normalize Confusion Matrix for Better Insights**

```python
# Normalized by true labels → shows recall per class
metrics_calc.plot_confusion_matrix(y_true, y_pred, normalize='true')

# Normalized by predicted labels → shows precision per class
metrics_calc.plot_confusion_matrix(y_true, y_pred, normalize='pred')
```

### 5. **Log Metrics to MLOps Platform**

```python
import json

# Export as JSON for MLflow, W&B, etc.
metrics_calc.save_metrics_to_file(metrics, format='json')

with open('metrics.json', 'r') as f:
    metrics_dict = json.load(f)
    mlflow.log_metrics(metrics_dict)
```

---

## Troubleshooting

### Issue: "Cannot convert type X to numpy array"

**Solution:** Ensure inputs are numpy arrays, lists, or tensors:

```python
y_true = np.array(y_true)  # Convert list to numpy
```

### Issue: "Length mismatch" error

**Solution:** Verify y_true and y_pred have the same length:

```python
assert len(y_true) == len(y_pred), "Mismatch!"
```

### Issue: Top-K accuracy returns None

**Solution:** Provide `y_pred_proba` (prediction probabilities):

```python
# Get probabilities from your model
y_pred_proba = model.predict_proba(X_test)

metrics = metrics_calc.compute_all_metrics(
    y_true, y_pred,
    y_pred_proba=y_pred_proba,  # Required for top-K
    top_k=3
)
```

---

## API Reference

### GestureClassificationMetrics

**Constructor:**

```python
GestureClassificationMetrics(
    class_names: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8)
)
```

**Main Methods:**

- `compute_all_metrics(y_true, y_pred, y_pred_proba, top_k, ...)` → Dict
- `compute_accuracy(y_true, y_pred)` → float
- `compute_precision_recall_f1(y_true, y_pred, average)` → Tuple
- `compute_confusion_matrix(y_true, y_pred, normalize)` → np.ndarray
- `compute_top_k_accuracy(y_true, y_pred_proba, k)` → float
- `plot_confusion_matrix(y_true, y_pred, normalize, save_path, ...)` → Figure
- `plot_per_class_metrics(y_true, y_pred, save_path, ...)` → Figure
- `print_summary(metrics, detailed)` → None
- `save_metrics_to_file(metrics, filepath, format)` → None

---

## Examples from Real Use Cases

### Use Case 1: Comparing Two Models

```python
from eva_recog_metrics import GestureClassificationMetrics

# Evaluate Model A
metrics_calc_a = GestureClassificationMetrics(
    class_names=gesture_classes,
    output_dir='./results/model_a'
)
metrics_a = metrics_calc_a.compute_all_metrics(y_true, y_pred_model_a)

# Evaluate Model B
metrics_calc_b = GestureClassificationMetrics(
    class_names=gesture_classes,
    output_dir='./results/model_b'
)
metrics_b = metrics_calc_b.compute_all_metrics(y_true, y_pred_model_b)

# Compare
print(f"Model A F1 (Macro): {metrics_a['f1_macro']:.4f}")
print(f"Model B F1 (Macro): {metrics_b['f1_macro']:.4f}")
```

### Use Case 2: Continuous Evaluation in Production

```python
import schedule

def evaluate_production_model():
    """Run evaluation on latest production data"""
    # Load latest predictions from database
    y_true, y_pred = load_from_db()

    metrics_calc = GestureClassificationMetrics(
        class_names=GESTURE_CLASSES,
        output_dir=f'./prod_metrics/{datetime.now().strftime("%Y%m%d")}'
    )

    metrics = metrics_calc.compute_all_metrics(y_true, y_pred)

    # Alert if accuracy drops
    if metrics['accuracy'] < 0.85:
        send_alert(f"⚠️ Accuracy dropped to {metrics['accuracy']:.2%}")

    # Save for dashboard
    metrics_calc.save_metrics_to_file(metrics, format='json')

# Run daily
schedule.every().day.at("02:00").do(evaluate_production_model)
```

---

## Contributing

Feel free to extend this module with:

- Additional metrics (AUC-ROC, PR curves, etc.)
- More visualization types
- Integration with specific MLOps tools
- Custom averaging schemes

---

## License

Part of the Hand Gesture Recognition project. See main repository for license details.

---

**Questions?** Check the inline documentation in `eva_recog_metrics.py` or run the demo!
