"""
Gesture Recognition Module
Recognizes specific hand gestures from landmark data
"""
import numpy as np


class GestureRecognizer:
    """Recognizes hand gestures from landmark positions"""
    
    def __init__(self):
        self.gesture_names = {
            'fist': 'Fist',
            'palm': 'Palm Open',
            'thumbs_up': 'Thumbs Up',
            'thumbs_down': 'Thumbs Down',
            'point': 'Pointing',
            'peace': 'Peace Sign',
            'ok': 'OK Sign',
            'swipe_left': 'Swipe Left',
            'swipe_right': 'Swipe Right',
            'swipe_up': 'Swipe Up',
            'swipe_down': 'Swipe Down',
            'pinch': 'Pinch',
            'open_pinch': 'Open Pinch', 
            'unknown': 'Unknown'
        }
        
        self.prev_position = None
        self.movement_threshold = 50  # pixels
        
    def recognize_gesture(self, fingers, landmark_list, distance_thumb_index=None):
        """
        Recognize gesture based on finger states and landmarks
        
        Args:
            fingers: List of finger states [thumb, index, middle, ring, pinky]
            landmark_list: List of landmark positions
            distance_thumb_index: Distance between thumb and index finger (for pinch detection)
            
        Returns:
            Gesture name string
        """
        if not fingers or len(fingers) != 5:
            return 'unknown'
        
        # Count fingers up
        fingers_count = sum(fingers)
        
        # Fist - all fingers down
        if fingers_count == 0 and distance_thumb_index > 20:
            return 'fist'
        
        # Palm open - all fingers up
        if fingers_count == 5:
            return 'palm'
        
        # Thumbs up - only thumb up
        if fingers == [1, 0, 0, 0, 0] and distance_thumb_index > 20:
            return 'thumbs_up'
        
        # Thumbs down - special case (would need orientation detection)
        # For now, we'll skip this as it requires more complex logic
        
        # Pointing - only index finger up
        if fingers == [0, 1, 0, 0, 0] and distance_thumb_index > 20 and distance_thumb_index < 120:
            return 'point'
        
        # Peace sign - index and middle fingers up
        if fingers == [0, 1, 1, 0, 0] and distance_thumb_index > 20:
            return 'peace'
        
        # OK sign - thumb and index close (pinch gesture)
        if distance_thumb_index is not None and distance_thumb_index < 13:
            return 'pinch'

        if distance_thumb_index is not None and distance_thumb_index > 150:
            return 'open_pinch'
        
        # Three fingers up
        if fingers_count == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            return 'ok'
        
        return 'unknown'
    
    def detect_swipe(self, landmark_list):
        """
        Detect swipe gestures based on hand movement
        
        Args:
            landmark_list: Current hand landmark positions
            
        Returns:
            Swipe direction or None
        """
        if not landmark_list or len(landmark_list) == 0:
            self.prev_position = None
            return None
        
        # Use wrist position (landmark 0) for movement tracking
        current_position = (landmark_list[0][1], landmark_list[0][2])
        
        if self.prev_position is None:
            self.prev_position = current_position
            return None
        
        # Calculate movement
        dx = current_position[0] - self.prev_position[0]
        dy = current_position[1] - self.prev_position[1]
        
        # Check if movement exceeds threshold
        if abs(dx) > self.movement_threshold or abs(dy) > self.movement_threshold:
            swipe = None
            
            # Determine direction
            if abs(dx) > abs(dy):
                if dx > 0:
                    swipe = 'swipe_right'
                else:
                    swipe = 'swipe_left'
            else:
                if dy > 0:
                    swipe = 'swipe_down'
                else:
                    swipe = 'swipe_up'
            
            self.prev_position = current_position
            return swipe
        
        self.prev_position = current_position
        return None
    
    def get_gesture_action(self, gesture, mode='general'):
        """
        Map gesture to action based on current mode
        
        Args:
            gesture: Recognized gesture name
            mode: Current operation mode (pdf, media, general)
            
        Returns:
            Action description string
        """
        actions = {
            'pdf': {
                'point': 'Scroll Up',
                'peace': 'Scroll Down',
                'swipe_left': 'Previous Page',
                'swipe_right': 'Next Page',
                'pinch': 'Zoom Out',
                'open_pinch': 'Zoom In',
                'fist': 'Stop',
                'palm': 'Reset View'
            },
            'media': {
                'palm': 'Play/Pause',
                'swipe_right': 'Next Track',
                'swipe_left': 'Previous Track',
                'swipe_up': 'Volume Up',
                'swipe_down': 'Volume Down',
                'fist': 'Stop',
                'thumbs_up': 'Like',
                'point': 'Select'
            },
            'general': {
                'point': 'Move Cursor',
                'pinch': 'Click',
                'palm': 'Release',
                'fist': 'Drag',
                'swipe_up': 'Scroll Up',
                'swipe_down': 'Scroll Down',
                'peace': 'Right Click'
            }
        }
        
        mode_actions = actions.get(mode, actions['general'])
        return mode_actions.get(gesture, 'No Action')