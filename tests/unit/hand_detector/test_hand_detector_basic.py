# tests/unit/test_hand_detector_basic.py
import sys
import os

# Add src to path so we can import HandDetector
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

def test_import():
    """Sanity check - can we import the module?"""
    try:
        from detectors.hand_detector import HandDetector
        assert True, "Import successful"
    except ImportError as e:
        assert False, f"Import failed: {e}"