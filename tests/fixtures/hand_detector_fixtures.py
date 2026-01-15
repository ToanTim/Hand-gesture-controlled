# tests/fixtures/hand_detector_fixtures.py
import numpy as np
from unittest.mock import Mock

def create_mock_frame(width=640, height=480, channels=3):
    """Create a dummy frame for testing"""
    return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)

def create_mock_hand_landmarks(hand_index=0, num_landmarks=21):
    """Create mock MediaPipe hand landmarks"""
    mock_landmarks = []
    mock_hand = Mock()
    
    # Create 21 mock landmarks (MediaPipe standard)
    for i in range(num_landmarks):
        landmark = Mock()
        landmark.x = 0.1 * i  # Spread them out
        landmark.y = 0.1 * i
        landmark.z = 0.0
        mock_landmarks.append(landmark)
    
    mock_hand.landmark = mock_landmarks
    return mock_hand

def create_mock_mp_results(multi_hand_landmarks=None, multi_handedness=None):
    """Create a mock MediaPipe results object"""
    mock_results = Mock()
    mock_results.multi_hand_landmarks = multi_hand_landmarks or []
    mock_results.multi_handedness = multi_handedness or []
    return mock_results

def create_sample_landmark_list(num_points=21):
    """Create a sample landmark list in the format [(id, x, y), ...]"""
    return [(i, i*10, i*10) for i in range(num_points)]