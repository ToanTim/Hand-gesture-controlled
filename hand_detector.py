"""
Hand Gesture Detection Module
Uses MediaPipe to detect hand landmarks and track hand movements
"""
import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """Detects hands and hand landmarks using MediaPipe"""
    
    def __init__(self, max_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        """
        Initialize hand detector
        
        Args:
            max_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
    def find_hands(self, frame, draw=True):
        """
        Detect hands in the frame
        
        Args:
            frame: Input frame from webcam
            draw: Whether to draw hand landmarks on frame
            
        Returns:
            Processed frame with drawings (if enabled)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        self.results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame
    
    def get_position(self, frame, hand_no=0):
        """
        Get positions of hand landmarks
        
        Args:
            frame: Input frame
            hand_no: Which hand (0 for first, 1 for second)
            
        Returns:
            List of landmark positions [(id, x, y), ...]
        """
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                
                h, w, c = frame.shape
                
                for id, landmark in enumerate(hand.landmark):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmark_list.append((id, cx, cy))
        
        return landmark_list
    
    def fingers_up(self, landmark_list):
        """
        Check which fingers are up
        
        Args:
            landmark_list: List of landmark positions
            
        Returns:
            List of 5 values (thumb, index, middle, ring, pinky) - 1 if up, 0 if down
        """
        if len(landmark_list) == 0:
            return []
        
        fingers = []
        tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        # Thumb - special case (check horizontal position)
        if landmark_list[tip_ids[0]][1] < landmark_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers - check if tip is above the pip joint
        for i in range(1, 5):
            if landmark_list[tip_ids[i]][2] < landmark_list[tip_ids[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def get_distance(self, p1, p2, landmark_list):
        """
        Calculate distance between two landmarks
        
        Args:
            p1: First point landmark id
            p2: Second point landmark id
            landmark_list: List of landmarks
            
        Returns:
            Distance in pixels, and center point coordinates
        """
        if len(landmark_list) < max(p1, p2) + 1:
            return 0, (0, 0)
        
        x1, y1 = landmark_list[p1][1], landmark_list[p1][2]
        x2, y2 = landmark_list[p2][1], landmark_list[p2][2]
        
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        return distance, (cx, cy)
    
    def close(self):
        """Release resources"""
        self.hands.close()
