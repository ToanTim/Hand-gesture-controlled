# Performance Evaluation Module - Implementation Summary

## ✅ Completed Implementation

A comprehensive performance evaluation module has been created for measuring and reporting computer vision model performance with both visual and text outputs.

---

## 📁 Files Created

### 1. Core Utilities

**Location:** `src/recognizers/utils/performance_metric.py`

**Purpose:** Reusable functions for performance measurement

**Features:**

- ✓ Inference latency measurement (ms/frame)
- ✓ FPS calculation and tracking
- ✓ Model load time measurement
- ✓ CPU utilization monitoring
- ✓ GPU utilization monitoring (when available)
- ✓ Memory usage tracking (RSS, VMS)
- ✓ Model size estimation
- ✓ Statistical analysis (mean, median, std, percentiles)
- ✓ Threshold checking with warnings
- ✓ Context manager for easy tracking (PerformanceTracker)

**Key Classes:**

- `PerformanceMetrics` - Metrics accumulator
- `PerformanceTracker` - Context manager for automatic tracking

**Key Functions:**

- `measure_model_load_time()` - Time model loading
- `measure_inference_latency()` - Time single inference
- `calculate_fps()` - Compute FPS from latencies
- `get_memory_usage()` - Get current memory stats
- `get_cpu_usage()` - Get current CPU stats
- `get_gpu_usage()` - Get current GPU stats (if available)
- `compute_statistics()` - Statistical analysis
- `check_performance_thresholds()` - Validate against thresholds

---

### 2. Evaluation Script

**Location:** `src/recognizers/evaluation/eva_recog_performance.py`

**Purpose:** Main evaluation script for running comprehensive performance tests

**Features:**

- ✓ Loads gesture recognition models (HandDetector + GestureRecognizer)
- ✓ Runs inference benchmark on configurable number of frames
- ✓ Supports multiple input sources (webcam, video file, synthetic frames)
- ✓ Collects real-time metrics during execution
- ✓ Generates visual reports with multiple plots
- ✓ Generates detailed text reports
- ✓ Saves metrics to JSON format

**Main Class:**

- `GesturePerformanceEvaluator` - Orchestrates complete evaluation

**Visual Outputs:**

1. **performance_report.png** - 6-panel dashboard:
   - Latency histogram with mean/P95 lines
   - FPS over time plot
   - Memory usage timeline
   - CPU utilization timeline
   - Latency box plot
   - Summary statistics table

2. **latency_per_frame.png** - Detailed latency analysis

**Text Outputs:**

1. **performance_report.txt** - Comprehensive report with:
   - Model load times
   - Latency statistics (mean, median, std, min, max, P50, P95, P99)
   - FPS metrics
   - Memory usage statistics
   - CPU/GPU utilization
   - Processing summary
   - Performance analysis with warnings

---

### 3. Orchestration Script

**Location:** `src/recognizers/evaluation/check_eva_recog_metrics.py`

**Purpose:** Run evaluation and save results in structured formats

**Features:**

- ✓ Executes eva_recog_performance.py
- ✓ Loads and processes generated metrics
- ✓ Saves results in CSV format (metrics_summary.csv)
- ✓ Saves results in JSON format (metrics.json)
- ✓ Generates quick summary report (summary.txt)
- ✓ Includes performance assessment (Excellent/Good/Acceptable/Poor)
- ✓ Provides recommendations for improvement
- ✓ Verifies all output files

**Main Class:**

- `PerformanceMetricsChecker` - Complete pipeline orchestration

**CSV Output Format:**

```csv
Timestamp,Metric Category,Metric Name,Value,Unit
2026-01-19 13:42,Inference Latency,Mean,25.43,ms
2026-01-19 13:42,FPS,Average FPS,39.32,fps
2026-01-19 13:42,Memory Usage,Max,289.43,MB
2026-01-19 13:42,CPU Usage,Mean,45.23,%
```

**Command-line Options:**

- `--results-dir` - Custom output directory
- `--load-only` - Process existing results without re-running

---

### 4. Quick Start Script

**Location:** `src/recognizers/evaluation/quick_start_evaluation.py`

**Purpose:** Simple interface for running evaluations

**Features:**

- ✓ User-friendly command-line interface
- ✓ Configurable parameters (frames, input source, output dir)
- ✓ Optional plot generation
- ✓ Summary-only mode
- ✓ Interactive prompts

**Usage Examples:**

```bash
# Basic usage
python quick_start_evaluation.py

# Custom frames
python quick_start_evaluation.py --frames 200

# Use webcam
python quick_start_evaluation.py --camera

# Use video file
python quick_start_evaluation.py --video path/to/video.mp4

# Summary only (fast)
python quick_start_evaluation.py --summary-only --no-plots
```

---

### 5. Documentation

**Location:** `src/recognizers/evaluation/PERFORMANCE_EVALUATION_GUIDE.md`

**Purpose:** Complete user guide and API documentation

**Contents:**

- Overview and architecture
- Quick start guide
- Metrics documentation
- Output file descriptions
- Usage examples
- Customization guide
- Troubleshooting
- CI/CD integration examples

---

### 6. Usage Examples

**Location:** `src/recognizers/utils/example_performance_usage.py`

**Purpose:** Demonstrate how to use performance metrics in custom code

**Examples Included:**

1. Basic performance tracking with context manager
2. Measure specific function latency
3. Track performance across multiple inferences
4. Monitor system resources
5. Check performance against thresholds
6. Measure model load time
7. Continuous performance monitoring

---

## 📊 Metrics Measured

### Load Time Metrics

- Detector load time (ms)
- Recognizer load time (ms)
- Total load time (ms)

### Inference Metrics

- Mean latency (ms)
- Median latency (ms)
- Standard deviation (ms)
- Min/Max latency (ms)
- P50, P95, P99 percentiles (ms)

### Throughput Metrics

- Average FPS
- FPS over time (sliding window)
- Min/Max FPS

### Resource Metrics

- Memory usage (RSS, VMS in MB)
- CPU utilization (process & system %)
- GPU utilization (% and memory, if available)

### Statistical Analysis

- Mean, median, std deviation
- Min, max values
- Percentiles (50th, 95th, 99th)

---

## 🎯 Performance Thresholds

Default warning thresholds:

- **Latency:** 50ms (warns if exceeded)
- **FPS:** 20 FPS (warns if below)
- **Memory:** 1000 MB (warns if exceeded)
- **CPU:** 80% (warns if exceeded)

Assessment levels:

- **Excellent:** Best performance tier
- **Good:** Acceptable performance
- **Acceptable:** Marginal performance
- **Poor:** Below acceptable standards

---

## 🚀 How to Use

### Quick Start (Recommended)

```bash
cd /home/toantim/ToanFolder/Hand-gesture-controlled
python src/recognizers/evaluation/quick_start_evaluation.py
```

### Full Pipeline with Results Saving

```bash
python src/recognizers/evaluation/check_eva_recog_metrics.py
```

### Direct Evaluation Script

```bash
python src/recognizers/evaluation/eva_recog_performance.py
```

### View Examples

```bash
python src/recognizers/utils/example_performance_usage.py
```

---

## 📦 Output Structure

```
src/recognizers/evaluation/results/
├── metrics.json                    # Complete metrics (JSON)
├── metrics_summary.csv             # Tabular metrics (CSV)
├── performance_report.txt          # Detailed report (TXT)
├── summary.txt                     # Quick summary (TXT)
└── plots/
    ├── performance_report.png      # 6-panel dashboard
    └── latency_per_frame.png       # Detailed latency plot
```

---

## 🔧 Customization Options

### Number of Frames

Edit in script or pass as parameter:

```python
num_frames=200  # Default: 100
```

### Input Source

Options:

- Synthetic frames (default, no camera needed)
- Webcam (`use_camera=True`)
- Video file (`video_path="path/to/video.mp4"`)

### Thresholds

Customize warning thresholds:

```python
thresholds={
    'max_latency_ms': 30.0,
    'min_fps': 25.0,
    'max_memory_mb': 800.0,
    'max_cpu_percent': 70.0
}
```

### Output Directory

Change results location:

```bash
--results-dir custom_output/
```

---

## 📚 Integration Examples

### In Your Code

```python
from src.recognizers.utils.performance_metric import PerformanceTracker

with PerformanceTracker("My Task"):
    result = my_inference_function()
# Automatically prints timing and memory delta
```

### CI/CD Pipeline

```bash
# Run evaluation
python src/recognizers/evaluation/check_eva_recog_metrics.py

# Check metrics
python -c "
import json
metrics = json.load(open('results/metrics.json'))
assert metrics['latency']['mean'] < 50, 'Latency regression!'
"
```

---

## ✨ Key Features

### Visual Reports

- ✅ Histogram plots with statistical markers
- ✅ Time series plots with filled areas
- ✅ Box plots for distribution analysis
- ✅ Summary tables embedded in figures
- ✅ Professional styling with seaborn
- ✅ High-resolution output (300 DPI)

### Text Reports

- ✅ Formatted ASCII tables
- ✅ Section-based organization
- ✅ Color-coded status indicators (✓, ⚠, ✗)
- ✅ Threshold validation
- ✅ Automatic recommendations

### Data Formats

- ✅ JSON for programmatic access
- ✅ CSV for spreadsheet analysis
- ✅ TXT for human readability

### Code Quality

- ✅ Comprehensive docstrings
- ✅ Type hints throughout
- ✅ Modular function design
- ✅ Error handling
- ✅ Context managers for safety
- ✅ Clean, readable code structure

---

## 🧪 Testing the Implementation

### Quick Test (No dependencies on models)

```bash
python src/recognizers/utils/example_performance_usage.py
```

This runs all 7 usage examples without requiring trained models.

### Full Evaluation Test

```bash
python src/recognizers/evaluation/check_eva_recog_metrics.py
```

This runs the complete pipeline with actual models.

---

## 📋 Dependencies

**Required:**

- numpy - Numerical operations
- matplotlib - Plotting
- seaborn - Statistical plots
- psutil - System monitoring
- opencv-python (cv2) - Video processing

**Optional:**

- nvidia-ml-py3 (pynvml) - GPU monitoring

**Install:**

```bash
pip install numpy matplotlib seaborn psutil opencv-python
pip install nvidia-ml-py3  # Optional for GPU
```

---

## 🎓 Learning Resources

1. **Start here:** `PERFORMANCE_EVALUATION_GUIDE.md`
2. **See examples:** `example_performance_usage.py`
3. **Quick test:** `quick_start_evaluation.py`
4. **Full pipeline:** `check_eva_recog_metrics.py`

---

## ✅ Requirements Met

### ✓ Requirement 1: utils/performance_metric.py

- [x] Inference latency measurement
- [x] FPS calculation
- [x] Model load time measurement
- [x] CPU/GPU utilization
- [x] Memory usage tracking
- [x] Model size utilities

### ✓ Requirement 2: eva_recog_performance.py

- [x] Loads models
- [x] Runs inference on dataset/video
- [x] Uses performance_metric.py functions
- [x] Generates visual reports (plots)
- [x] Generates text reports (console + file)

### ✓ Requirement 3: check_eva_recog_metrics.py

- [x] Calls eva_recog_performance.py
- [x] Saves metrics in CSV format
- [x] Saves metrics in JSON format
- [x] Structured output in results/ folder
- [x] Includes all required metrics

### ✓ Visual Report Requirements

- [x] Latency histogram
- [x] FPS over time plot
- [x] Memory usage plot
- [x] CPU/GPU utilization plot

### ✓ Text Report Requirements

- [x] All metric values displayed
- [x] Summary table included
- [x] Warnings for threshold violations
- [x] Console output
- [x] Saved to file

### ✓ Code Quality

- [x] Clean structure
- [x] Modular functions
- [x] Comprehensive comments
- [x] Type hints
- [x] Error handling

---

## 🏆 Summary

A complete, production-ready performance evaluation module has been implemented with:

- **3 main scripts** (evaluation, checker, quick-start)
- **1 utility module** (performance_metric.py)
- **1 example script** (7 usage examples)
- **1 comprehensive guide** (PERFORMANCE_EVALUATION_GUIDE.md)
- **Multiple output formats** (JSON, CSV, TXT, PNG)
- **Professional visualizations** (6-panel dashboard + detailed plots)
- **Comprehensive metrics** (load time, latency, FPS, memory, CPU, GPU)
- **Smart analysis** (statistics, thresholds, recommendations)

The module is ready for immediate use and can be easily integrated into existing workflows!

---

**Created:** 2026-01-19  
**Location:** `/home/toantim/ToanFolder/Hand-gesture-controlled/src/recognizers/`  
**Status:** ✅ Complete and ready for use
