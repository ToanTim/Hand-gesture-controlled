"""
Data loaders for gesture recognition datasets.

This module contains functions for loading and preprocessing gesture recognition data
from various sources, particularly the HAGRID dataset.
"""

import os
import json
import warnings
from typing import List, Dict, Any

# Check for cv2 availability
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available. Install with: pip install opencv-python")


def load_hagrid_samples(data_dir: str, labels: List[str], samples_per_label: int = 50) -> List[Dict[str, Any]]:
    """
    Load samples from HAGRID dataset for specific labels.
    
    This function loads gesture image samples from the HAGRID dataset, processes them
    through a hand detector, and extracts relevant features (landmarks, finger states,
    and distances) for gesture recognition evaluation.
    
    Args:
        data_dir: Path to hagrid-sample-30k-384p directory
        labels: List of gesture labels to load (e.g., ['fist', 'palm', 'like', 'one', 'peace', 'ok'])
        samples_per_label: Number of samples to load per label (default 50)
    
    Returns:
        List of test data dicts with the following keys:
            - 'landmarks': Hand landmark coordinates
            - 'fingers': Finger up/down states
            - 'dist': Distance between thumb and index finger
            - 'label': Gesture label
            - 'img_id': Image identifier
    
    Raises:
        ImportError: If OpenCV is not installed
        
    Example:
        >>> data_dir = './data/hagrid-sample-30k-384p'
        >>> labels = ['fist', 'palm', 'like']
        >>> test_data = load_hagrid_samples(data_dir, labels, samples_per_label=50)
        >>> print(f"Loaded {len(test_data)} samples")
    """
    from detectors.hand_detector import HandDetector
    
    if not CV2_AVAILABLE:
        raise ImportError(
            "OpenCV is required for loading HAGRID samples. "
            "Install with: pip install opencv-python"
        )
    
    test_data = []
    ann_dir = os.path.join(data_dir, 'ann_train_val')
    img_dir = os.path.join(data_dir, 'hagrid_30k')
    
    hand_detector = HandDetector()
    
    for label in labels:
        ann_file = os.path.join(ann_dir, f'{label}.json')
        
        if not os.path.exists(ann_file):
            print(f"Warning: Annotation file not found for label '{label}'")
            continue
        
        print(f"Loading samples for '{label}'...")
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        count = 0
        for img_id, data in annotations.items():
            if count >= samples_per_label:
                break
            
            # Find the image file
            label_dir = os.path.join(img_dir, f'train_val_{label}')
            img_path = os.path.join(label_dir, f'{img_id}.jpg')
            
            if not os.path.exists(img_path):
                continue
            
            try:
                # Read image and detect hand landmarks
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                
                frame = hand_detector.find_hands(frame, draw=False)
                landmark_list = hand_detector.get_position(frame)
                
                if not landmark_list:
                    continue
                
                # Get finger states
                fingers = hand_detector.fingers_up(landmark_list)
                
                # Get distance between thumb and index
                distance, _ = hand_detector.get_distance(4, 8, landmark_list)
                
                # Add to test data
                test_data.append({
                    'landmarks': landmark_list,
                    'fingers': fingers,
                    'dist': distance,
                    'label': label,
                    'img_id': img_id
                })
                
                count += 1
            
            except Exception as e:
                print(f"Error processing {img_id}: {e}")
                continue
        
        print(f"Loaded {count} samples for '{label}'")
    
    hand_detector.close()
    return test_data
