# tests/unit/test_hand_detector_simple.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from unittest.mock import Mock, patch

def test_detector_parameters():
    """Simplest test - just check parameters are stored"""
    # Mock EVERYTHING from mediapipe and cv2
    with patch('detectors.hand_detector.mp.solutions.hands') as mock_hands, \
         patch('detectors.hand_detector.mp.solutions.drawing_utils') as mock_draw, \
         patch('detectors.hand_detector.mp.solutions.drawing_styles') as mock_styles:
        
        # Mock the Hands class constructor
        mock_hands_instance = Mock()
        mock_hands.Hands.return_value = mock_hands_instance
        
        from detectors.hand_detector import HandDetector
        
        # Test
        detector = HandDetector(max_hands=3, min_detection_confidence=0.9)
        
        # Just check the parameters were stored
        assert detector.max_hands == 3
        assert detector.min_detection_confidence == 0.9
        assert detector.min_tracking_confidence == 0.5  # default
        
        # Check hands was created
        assert detector.hands is not None