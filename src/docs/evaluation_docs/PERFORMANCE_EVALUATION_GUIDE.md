# Performance Evaluation Module

## Overview

This module provides comprehensive performance and efficiency evaluation for the gesture recognition system. It measures and reports various metrics including inference latency, FPS, CPU/GPU utilization, and memory usage with both visual and text outputs.

## Files Structure

```
src/recognizers/
├── utils/
│   └── performance_metric.py          # Reusable metrics functions
└── evaluation/
    ├── eva_recog_performance.py       # Main evaluation script
    ├── check_eva_recog_metrics.py     # Orchestration & results saver
    └── results/                       # Output directory (auto-created)
        ├── metrics.json               # Complete metrics in JSON
        ├── metrics_summary.csv        # Summary in CSV format
        ├── performance_report.txt     # Detailed text report
        ├── summary.txt                # Quick summary & recommendations
        └── plots/                     # Visual reports
            ├── performance_report.png
            └── latency_per_frame.png
```

## Quick Start

### Basic Usage

Run the complete evaluation pipeline:

```bash
cd /home/toantim/ToanFolder/Hand-gesture-controlled
python src/recognizers/evaluation/check_eva_recog_metrics.py
```

This will:

1. Load the gesture recognition models
2. Run inference benchmark (100 frames by default)
3. Measure all performance metrics
4. Generate visual and text reports
5. Save results in structured formats (JSON + CSV)

### Advanced Usage

Run evaluation script directly (for more control):

```bash
python src/recognizers/evaluation/eva_recog_performance.py
```

Process existing results only (skip re-evaluation):

```bash
python src/recognizers/evaluation/check_eva_recog_metrics.py --load-only
```

Custom results directory:

```bash
python src/recognizers/evaluation/check_eva_recog_metrics.py --results-dir custom_results/
```

## Metrics Measured

### 1. Model Load Time

- Detector load time (ms)
- Recognizer load time (ms)
- Total load time (ms)

### 2. Inference Latency

- Mean, median, std deviation
- Min, max values
- Percentiles (P50, P95, P99)

### 3. Frames Per Second (FPS)

- Average FPS
- FPS over time (sliding window)
- Min/max FPS

### 4. Memory Usage

- Resident Set Size (RSS)
- Mean, min, max memory
- Memory usage over time

### 5. CPU Utilization

- Process CPU usage (%)
- Mean, min, max CPU
- CPU usage over time

### 6. GPU Utilization (if available)

- GPU utilization (%)
- GPU memory usage
- Mean, min, max values

## Output Files

### 1. Visual Reports

**performance_report.png** - Comprehensive dashboard with:

- Latency histogram
- FPS over time
- Memory usage plot
- CPU utilization plot
- Latency box plot
- Summary statistics table

**latency_per_frame.png** - Detailed latency analysis per frame

### 2. Text Reports

**performance_report.txt** - Detailed report including:

- All metric values with statistics
- Processing summary
- Threshold warnings and errors
- Performance analysis

**summary.txt** - Quick summary with:

- Key metrics
- Performance assessment (Excellent/Good/Acceptable/Poor)
- Recommendations for improvement

### 3. Data Files

**metrics.json** - Complete metrics in JSON format:

```json
{
  "load_time": {...},
  "latency": {...},
  "fps": {...},
  "memory": {...},
  "cpu": {...},
  "gpu": {...}
}
```

**metrics_summary.csv** - Tabular format:

```csv
Timestamp,Metric Category,Metric Name,Value,Unit
2026-01-19 13:42,Inference Latency,Mean,25.43,ms
2026-01-19 13:42,FPS,Average FPS,39.32,fps
...
```

## Using the Performance Metrics Library

### In Your Own Code

```python
from src.recognizers.utils.performance_metric import (
    PerformanceMetrics,
    measure_inference_latency,
    get_memory_usage,
    PerformanceTracker
)

# Track performance with context manager
with PerformanceTracker("My Task"):
    # Your code here
    result = my_function()

# Measure specific function latency
latency_ms, result = measure_inference_latency(my_function, arg1, arg2)

# Get current memory usage
memory = get_memory_usage()
print(f"Current memory: {memory['rss']:.2f} MB")

# Use metrics accumulator
metrics = PerformanceMetrics()
for frame in frames:
    latency, result = measure_inference_latency(process_frame, frame)
    metrics.latencies.append(latency)
    metrics.timestamps.append(time.time())
```

### Available Functions

- `measure_model_load_time()` - Measure model loading time
- `measure_inference_latency()` - Measure single inference latency
- `calculate_fps()` - Calculate FPS from latencies
- `calculate_fps_over_time()` - FPS over sliding window
- `get_memory_usage()` - Current memory statistics
- `get_cpu_usage()` - Current CPU statistics
- `get_gpu_usage()` - Current GPU statistics (if available)
- `compute_statistics()` - Statistical metrics from list
- `check_performance_thresholds()` - Check against thresholds

## Performance Thresholds

Default thresholds for warnings:

- **Max Latency**: 50ms
- **Min FPS**: 20 FPS
- **Max Memory**: 1000 MB
- **Max CPU**: 80%

The system will generate warnings if these are exceeded.

## Customization

### Modify Number of Frames

Edit `eva_recog_performance.py`:

```python
benchmark_results = evaluator.run_inference_benchmark(
    num_frames=200,  # Change this
    use_camera=False,
    video_path=None
)
```

### Use Webcam or Video File

```python
# Use webcam
benchmark_results = evaluator.run_inference_benchmark(
    num_frames=100,
    use_camera=True,
    video_path=None
)

# Use video file
benchmark_results = evaluator.run_inference_benchmark(
    num_frames=100,
    use_camera=False,
    video_path="path/to/video.mp4"
)
```

### Adjust Thresholds

In `eva_recog_performance.py`:

```python
report = evaluator.generate_text_report(
    metrics,
    thresholds={
        'max_latency_ms': 30.0,    # Stricter
        'min_fps': 25.0,            # Higher
        'max_memory_mb': 800.0,     # Lower
        'max_cpu_percent': 70.0     # Lower
    }
)
```

## Dependencies

Required packages:

- `numpy` - Numerical computations
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `psutil` - System/process utilities
- `cv2` (opencv-python) - Video processing
- `pynvml` (optional) - NVIDIA GPU monitoring

Install missing dependencies:

```bash
pip install numpy matplotlib seaborn psutil opencv-python
pip install nvidia-ml-py3  # Optional, for GPU monitoring
```

## Troubleshooting

### Issue: GPU metrics not available

**Solution**: Install `nvidia-ml-py3`:

```bash
pip install nvidia-ml-py3
```

Or the evaluation will continue without GPU metrics.

### Issue: Import errors

**Solution**: Ensure you're running from the project root:

```bash
cd /home/toantim/ToanFolder/Hand-gesture-controlled
python src/recognizers/evaluation/check_eva_recog_metrics.py
```

### Issue: No camera available

**Solution**: The system will automatically generate synthetic frames for testing if no camera/video is available.

## Integration with CI/CD

You can integrate this into your CI/CD pipeline to track performance over time:

```bash
#!/bin/bash
# run_performance_tests.sh

# Run evaluation
python src/recognizers/evaluation/check_eva_recog_metrics.py

# Check if latency exceeds threshold
MEAN_LATENCY=$(python -c "import json; print(json.load(open('src/recognizers/evaluation/results/metrics.json'))['latency']['mean'])")

if (( $(echo "$MEAN_LATENCY > 50" | bc -l) )); then
    echo "ERROR: Latency regression detected!"
    exit 1
fi

echo "Performance tests passed!"
```

## Example Output

```
================================================================================
GESTURE RECOGNITION PERFORMANCE EVALUATION REPORT
================================================================================

MODEL LOAD TIME
--------------------------------------------------------------------------------
  Detector Load Time:          125.43 ms
  Recognizer Load Time:         45.21 ms
  Total Load Time:             170.64 ms

INFERENCE LATENCY
--------------------------------------------------------------------------------
  Mean:           25.43 ms
  Median:         24.87 ms
  Std Dev:         3.21 ms
  P95:            30.12 ms
  P99:            32.45 ms

FRAMES PER SECOND (FPS)
--------------------------------------------------------------------------------
  Average FPS:       39.32

MEMORY USAGE
--------------------------------------------------------------------------------
  Mean:          245.67 MB
  Max:           289.43 MB

CPU UTILIZATION
--------------------------------------------------------------------------------
  Mean:           45.23 %
  Max:            67.89 %

PERFORMANCE ANALYSIS
--------------------------------------------------------------------------------
  ✓ All metrics within acceptable thresholds
  ℹ Latency is within acceptable range (25.43ms)
  ℹ FPS is within acceptable range (39.32)
  ℹ Memory usage is within acceptable range (289.43MB)
  ℹ CPU usage is within acceptable range (67.89%)
```

## Contributing

When adding new metrics:

1. Add the measurement function to `performance_metric.py`
2. Update `eva_recog_performance.py` to collect the metric
3. Update `check_eva_recog_metrics.py` to save it to CSV
4. Update this README with documentation

## License

This module is part of the Hand Gesture Control project.
