# Quick Reference - Performance Evaluation

## 🚀 Quick Start

```bash
# Basic evaluation (recommended)
python check_eva_recog_metrics.py

# Or use the quick start script
python quick_start_evaluation.py

# Test installation
python test_performance_installation.py
```

## 📁 Files in This Directory

| File                               | Purpose                                  |
| ---------------------------------- | ---------------------------------------- |
| `eva_recog_performance.py`         | Main evaluation script - runs benchmarks |
| `check_eva_recog_metrics.py`       | Orchestrator - saves results to CSV/JSON |
| `quick_start_evaluation.py`        | Easy-to-use interface with options       |
| `test_performance_installation.py` | Verify installation                      |
| `PERFORMANCE_EVALUATION_GUIDE.md`  | Complete documentation                   |
| `IMPLEMENTATION_SUMMARY.md`        | Technical summary                        |
| `results/`                         | Output directory (auto-created)          |

## 📊 What Gets Measured

- ⏱️ Inference latency (ms)
- 🎬 FPS (frames per second)
- 💾 Memory usage (MB)
- 🖥️ CPU utilization (%)
- 🎮 GPU utilization (% - if available)
- 📦 Model load time (ms)

## 📈 Output Files

```
results/
├── metrics.json              # Full metrics
├── metrics_summary.csv       # CSV format
├── performance_report.txt    # Detailed text report
├── summary.txt               # Quick summary
└── plots/
    ├── performance_report.png    # 6-panel dashboard
    └── latency_per_frame.png     # Latency details
```

## 🔧 Requirements

```bash
pip install numpy matplotlib seaborn psutil opencv-python
pip install nvidia-ml-py3  # Optional for GPU monitoring
```

## 📚 More Info

See `PERFORMANCE_EVALUATION_GUIDE.md` for complete documentation.
