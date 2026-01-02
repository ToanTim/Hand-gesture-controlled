"""
Main Application
Hand Gesture Controlled Interface
"""
import cv2
import time
from hand_detector import HandDetector
from gesture_recognizer import GestureRecognizer
from pdf_controller import PDFController
from media_controller import MediaController
from mouse_controller import MouseController


class HandGestureApp:
    """Main application for hand gesture control"""
    
    def __init__(self, mode='general'):
        """
        Initialize the application
        
        Args:
            mode: Control mode - 'general', 'pdf', or 'media'
        """
        self.mode = mode
        self.running = False
        
        # Initialize components
        self.detector = HandDetector(max_hands=1)
        self.recognizer = GestureRecognizer()
        
        # Initialize controllers
        self.pdf_controller = PDFController()
        self.media_controller = MediaController()
        self.mouse_controller = MouseController()
        
        # Video capture
        self.cap = None
        
        # UI settings
        self.fps = 0
        self.prev_time = 0
        
        # Current gesture tracking
        self.current_gesture = None
        self.current_action = None
        self.current_distance = None
        self.finger_count = 0
        self.zoom_in_active = False
        
    def initialize_camera(self, camera_id=0):
        """Initialize webcam"""
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        return True
    
    def calculate_fps(self):
        """Calculate and return FPS"""
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        return int(fps)
    
    def draw_ui(self, frame):
        """Draw UI elements on frame"""
        h, w, _ = frame.shape
        
        # Draw mode
        cv2.rectangle(frame, (10, 10), (300, 100), (50, 50, 50), -1)
        cv2.putText(frame, f"Mode: {self.mode.upper()}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw gesture
        if self.current_gesture:
            gesture_name = self.recognizer.gesture_names.get(self.current_gesture, 'Unknown')
            cv2.putText(frame, f"Gesture: {gesture_name}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw action
        if self.current_action:
            cv2.rectangle(frame, (10, h - 60), (300, h - 10), (50, 50, 50), -1)
            cv2.putText(frame, f"Action: {self.current_action}", (20, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps}", (w - 120, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw distance
        if self.current_distance is not None:
            cv2.putText(frame, f"Dist: {int(self.current_distance)}", (w - 150, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Fingers: {self.finger_count}", (w - 170, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        # Draw instructions
        cv2.putText(frame, "Press 'q' to quit | 'm' to change mode", (10, h - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def process_gesture(self, gesture, landmark_list, distance=None):
        """Process recognized gesture and perform action"""
        action = None
        
        if self.mode == 'pdf':
            if distance is not None and distance > 100:
                self.zoom_in_active = True
                action = self.pdf_controller.handle_gesture('zoom_in')
            elif self.zoom_in_active and distance is not None and distance <= 100:
                self.zoom_in_active = False
            if action is None:
                action = self.pdf_controller.handle_gesture(gesture)
        elif self.mode == 'media':
            action = self.media_controller.handle_gesture(gesture)
        elif self.mode == 'general':
            action = self.mouse_controller.handle_gesture(gesture, landmark_list)
        
        return action
    
    def change_mode(self):
        """Cycle through control modes"""
        modes = ['general', 'pdf', 'media']
        current_index = modes.index(self.mode)
        self.mode = modes[(current_index + 1) % len(modes)]
        print(f"Switched to {self.mode.upper()} mode")
    
    def run(self):
        """Main application loop"""
        print("Starting Hand Gesture Control Application...")
        print(f"Mode: {self.mode.upper()}")
        print("Press 'q' to quit, 'm' to change mode")
        
        self.initialize_camera()
        self.running = True
        self.prev_time = time.time()
        
        try:
            while self.running:
                success, frame = self.cap.read()
                
                if not success:
                    print("Failed to read from webcam")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect hands
                frame = self.detector.find_hands(frame, draw=True)
                
                # Get hand landmarks
                landmark_list = self.detector.get_position(frame)
                
                # Process gestures if hand detected
                if landmark_list:
                    # Get finger states
                    fingers = self.detector.fingers_up(landmark_list)
                    self.finger_count = fingers.count(1)
                    # Get distance between thumb and index for pinch detection
                    distance, _ = self.detector.get_distance(4, 8, landmark_list)
                    self.current_distance = distance
                    
                    # Recognize static gesture
                    gesture = self.recognizer.recognize_gesture(fingers, landmark_list, distance)
                    
                    # Detect swipe gestures
                    swipe = self.recognizer.detect_swipe(landmark_list)
                    if swipe:
                        gesture = swipe
                    
                    # Update current gesture
                    self.current_gesture = gesture
                    
                    # Process gesture and get action
                    action = self.process_gesture(gesture, landmark_list, distance)
                    if action:
                        self.current_action = action
                
                # Calculate FPS
                self.fps = self.calculate_fps()
                
                # Draw UI
                frame = self.draw_ui(frame)
                
                # Display frame
                cv2.imshow('Hand Gesture Control', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    self.running = False
                elif key == ord('m'):
                    self.change_mode()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        print("Cleanup complete")


def main():
    """Main entry point"""
    import sys
    
    mode = 'general'
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ['general', 'pdf', 'media']:
            print(f"Unknown mode: {mode}")
            print("Available modes: general, pdf, media")
            print("Usage: python main.py [mode]")
            sys.exit(1)
    
    app = HandGestureApp(mode=mode)
    app.run()


if __name__ == "__main__":
    main()
