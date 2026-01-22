# Installation Guide

## Quick Start

To install the package in development mode (recommended for development):

```bash
# Navigate to project root
cd /home/toantim/ToanFolder/Hand-gesture-controlled

# Install in editable mode
pip install -e .
```

This will install the package and make all modules importable from anywhere.

## Running the Evaluation Module

After installation, you can run the evaluation module in multiple ways:

### Option 1: As a Python module (recommended)

```bash
# Run with HAGRID dataset evaluation
python -m src.recognizers.evaluation.eva_recog_metrics

# Run demo mode
python -m src.recognizers.evaluation.eva_recog_metrics --demo
```

### Option 2: Direct execution

```bash
# Navigate to the src directory
cd src/recognizers/evaluation
python eva_recog_metrics.py
```

### Option 3: Import and use in your code

```python
from recognizers.evaluation.eva_recog_metrics import evaluate_recognizer
from rdataecognizers.loaders import load_hagrid_samples
from recognizers.utils.metrics import GestureClassificationMetrics

# Your code here
```

## Installing with Optional Dependencies

### For PyTorch support:

```bash
pip install -e ".[torch]"
```

### For TensorFlow support:

```bash
pip install -e ".[tensorflow]"
```

### For development tools:

```bash
pip install -e ".[dev]"
```

### Install everything:

```bash
pip install -e ".[dev,torch,tensorflow]"
```

## Verifying Installation

```bash
# Check if package is installed
pip list | grep hand-gesture-control

# Test imports
python -c "from recognizers.utils.metrics import GestureClassificationMetrics; print('Success!')"
python -c "from recognizers.data_tils.loaders import load_hagrid_samples; print('Success!')"
```

## Troubleshooting

### ModuleNotFoundError

If you still get `ModuleNotFoundError`, make sure:

1. You've installed the package: `pip install -e .`
2. You're using the correct Python environment
3. The `src` directory is in your Python path

### Alternative: Add src to PYTHONPATH

If you don't want to install the package, you can add the src directory to PYTHONPATH:

```bash
# For current session only
export PYTHONPATH="${PYTHONPATH}:/home/toantim/ToanFolder/Hand-gesture-controlled/src"

# Or add to ~/.bashrc or ~/.zshrc for permanent effect
echo 'export PYTHONPATH="${PYTHONPATH}:/home/toantim/ToanFolder/Hand-gesture-controlled/src"' >> ~/.zshrc
source ~/.zshrc
```

## Project Structure After Refactoring

```
Hand-gesture-controlled/
├── pyproject.toml           # Modern Python project configuration
├── setup.py                 # Backward compatible setup script
├── README.md
├── requirements.txt
├── INSTALL.md              # This file
│
├── src/                    # Main package (installed as -e)
│   ├── __init__.py         # Package initialization
│   │
│   ├── recognizers/        # Gesture recognition logic
│   │   ├── __init__.py
│   │   │
│   │   ├── data/           # Data loading utilities
│   │   │   ├── __init__.py
│   │   │   └── loaders.py  # HAGRID and other data loaders
│   │   │
│   │   ├── utils/          # Metrics and utilities
│   │   │   ├── __init__.py
│   │   │   └── metrics.py  # Comprehensive metrics module
│   │   │
│   │   ├── gesture_recognizer.py
│   │   ├── gesture_model.py
│   │   ├── gesture_train.py
│   │   └── evaluation/
│   │       ├── eva_recog_metrics.py  # High-level evaluation
│   │       └── METRICS_USAGE.md
│   │
│   ├── detectors/          # Hand detection
│   │   └── hand_detector.py
│   │
│   ├── controllers/        # Control integrations
│   │   ├── media_controller.py
│   │   ├── mouse_controller.py
│   │   └── pdf_controller.py
│   │
│   └── main.py            # Entry point
│
├── tests/                 # Test suite
│   ├── unit/
│   └── integration/
│
├── config/               # Configuration files
│   └── config.json
│
└── docs/                # Documentation and results
    └── evaluation_result/
```

## Next Steps

After successful installation:

1. Run the demo: `python -m src.recognizers.evaluation.eva_recog_metrics --demo`
2. Check the generated outputs in `./demo_output/`
3. Review the comprehensive documentation in `METRICS_USAGE.md`
4. Start using the refactored modules in your code!
