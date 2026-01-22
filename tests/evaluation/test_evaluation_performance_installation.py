"""
Test Installation and Basic Functionality

This script verifies that all performance evaluation components
are properly installed and can be imported.

Run this to verify your installation:
    python test_performance_installation.py
"""

import sys
from pathlib import Path

# Change to project root directory
import os
script_dir = Path(__file__).parent.parent.parent.parent
os.chdir(script_dir)


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        print("  ✓ Importing performance_metric...")
        sys.path.insert(0, str(script_dir))
        from src.recognizers.utils.performance_metric import (
            PerformanceMetrics,
            PerformanceTracker,
            measure_inference_latency,
            calculate_fps,
            get_memory_usage,
            get_cpu_usage,
            compute_statistics
        )
        
        print("  ✓ Importing eva_recog_performance...")
        from src.recognizers.evaluation.eva_recog_performance import (
            GesturePerformanceEvaluator
        )
        
        print("  ✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are installed"""
    print("\nTesting dependencies...")
    
    missing = []
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - MISSING")
        missing.append("numpy")
    
    try:
        import matplotlib
        print("  ✓ matplotlib")
    except ImportError:
        print("  ✗ matplotlib - MISSING")
        missing.append("matplotlib")
    
    try:
        import seaborn
        print("  ✓ seaborn")
    except ImportError:
        print("  ✗ seaborn - MISSING")
        missing.append("seaborn")
    
    try:
        import psutil
        print("  ✓ psutil")
    except ImportError:
        print("  ✗ psutil - MISSING")
        missing.append("psutil")
    
    try:
        import cv2
        print("  ✓ opencv-python (cv2)")
    except ImportError:
        print("  ✗ opencv-python (cv2) - MISSING")
        missing.append("opencv-python")
    
    # Optional
    try:
        import pynvml
        print("  ✓ nvidia-ml-py3 (pynvml) - OPTIONAL")
    except ImportError:
        print("  ℹ nvidia-ml-py3 (pynvml) - not installed (optional for GPU)")
    
    if missing:
        print(f"\n  ⚠ Missing {len(missing)} required package(s)")
        print(f"\n  Install with: pip install {' '.join(missing)}")
        return False
    else:
        print("  ✓ All required dependencies installed!")
        return True


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from src.recognizers.utils.performance_metric import (
            PerformanceMetrics,
            get_memory_usage,
            compute_statistics
        )
        
        # Test PerformanceMetrics
        metrics = PerformanceMetrics()
        metrics.latencies = [10.0, 15.0, 12.0, 18.0, 14.0]
        print("  ✓ PerformanceMetrics instantiation")
        
        # Test get_memory_usage
        memory = get_memory_usage()
        assert 'rss' in memory
        print(f"  ✓ get_memory_usage() - Current memory: {memory['rss']:.2f}MB")
        
        # Test compute_statistics
        stats = compute_statistics(metrics.latencies)
        assert 'mean' in stats
        print(f"  ✓ compute_statistics() - Mean: {stats['mean']:.2f}")
        
        print("  ✓ Basic functionality working!")
        return True
        
    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist"""
    print("\nChecking file structure...")
    script_dir
    base_path = Path(__file__).parent.parent.parent
    
    files_to_check = [
        "src/recognizers/utils/performance_metric.py",
        "src/recognizers/utils/example_performance_usage.py",
        "src/recognizers/evaluation/eva_recog_performance.py",
        "src/recognizers/evaluation/quick_start_evaluation.py",
        "src/docs/evaluation_docs/PERFORMANCE_EVALUATION_GUIDE.md",
        "src/docs/evaluation_docs/IMPLEMENTATION_SUMMARY.md",
    ]
    
    all_exist = True
    for file_path in files_to_check:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_exist = False
    
    if all_exist:
        print("  ✓ All files present!")
    else:
        print("  ⚠ Some files missing")
    
    return all_exist


def main():
    """Run all tests"""
    print("="*80)
    print("PERFORMANCE EVALUATION MODULE - INSTALLATION TEST")
    print("="*80)
    print()
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test dependencies
    results.append(("Dependencies", test_dependencies()))
    
    # Test basic functionality
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Test file structure
    results.append(("File Structure", test_file_structure()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Installation is complete!")
        print("\nYou can now run:")
        print("  python src/recognizers/evaluation/quick_start_evaluation.py")
    else:
        print("⚠ SOME TESTS FAILED - Please check errors above")
        print("\nIf dependencies are missing, install them with:")
        print("  pip install numpy matplotlib seaborn psutil opencv-python")
    print("="*80)
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
