# tests/unit/test_hand_detector_init.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

import pytest
from unittest.mock import Mock, patch

class TestHandDetectorInit:
    """Test the initialization of HandDetector"""
    
    def test_init_default_params(self):
        """Test initialization with default parameters"""
        # We need to mock mediapipe to avoid actual initialization
        with patch('detectors.hand_detector.mp.solutions.hands') as mock_hands:
            # Mock the Hands class
            mock_hands_instance = Mock()
            mock_hands.Hands.return_value = mock_hands_instance
            
            from detectors.hand_detector import HandDetector
            
            # Create detector with default params
            detector = HandDetector()
            
            # Verify parameters
            assert detector.max_hands == 2
            assert detector.min_detection_confidence == 0.7
            assert detector.min_tracking_confidence == 0.5
            assert detector.hands == mock_hands_instance
            
            # Verify MediaPipe was called with correct params
            mock_hands.Hands.assert_called_once_with(
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters"""
        with patch('detectors.hand_detector.mp.solutions.hands') as mock_hands:
            mock_hands_instance = Mock()
            mock_hands.Hands.return_value = mock_hands_instance
            
            from detectors.hand_detector import HandDetector
            
            # Create detector with custom params
            detector = HandDetector(
                max_hands=1,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.6
            )
            
            # Verify parameters
            assert detector.max_hands == 1
            assert detector.min_detection_confidence == 0.8
            assert detector.min_tracking_confidence == 0.6
            
            # Verify MediaPipe was called with custom params
            mock_hands.Hands.assert_called_once_with(
                max_num_hands=1,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.6
            )