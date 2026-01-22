#!/usr/bin/env python3
"""
Quick test script to verify the refactored module structure.
Run this after installing the package with: pip install -e .
"""

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from recognizers.utils.metrics import GestureClassificationMetrics
        print("✓ GestureClassificationMetrics imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import GestureClassificationMetrics: {e}")
        return False
    
    try:
        from recognizers.data_utils.loaders import load_hagrid_samples
        print("✓ load_hagrid_samples imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import load_hagrid_samples: {e}")
        return False
    
    try:
        from recognizers.evaluation.eva_recog_metrics import evaluate_recognizer, demo_metrics_module
        print("✓ Evaluation functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluation functions: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_basic_functionality():
    """Test basic metrics functionality."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from recognizers.utils.metrics import GestureClassificationMetrics
        
        # Create simple test data
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])
        
        # Initialize metrics calculator
        metrics_calc = GestureClassificationMetrics(
            class_names=['class_0', 'class_1', 'class_2']
        )
        
        # Compute metrics
        metrics = metrics_calc.compute_all_metrics(y_true, y_pred)
        
        print(f"✓ Computed accuracy: {metrics['accuracy']:.4f}")
        print(f"✓ Computed macro F1: {metrics['f1_macro']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*70)
    print(" Module Structure Verification ".center(70, "="))
    print("="*70 + "\n")
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n" + "="*70)
        print(" FAILED: Please run 'pip install -e .' first ".center(70, "="))
        print("="*70)
        exit(1)
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    if functionality_ok:
        print("\n" + "="*70)
        print(" SUCCESS: All tests passed! ".center(70, "="))
        print("="*70)
        print("\nYou can now run:")
        print("  python -m src.recognizers.evaluation.eva_recog_metrics --demo")
        exit(0)
    else:
        print("\n" + "="*70)
        print(" FAILED: Functionality test failed ".center(70, "="))
        print("="*70)
        exit(1)
