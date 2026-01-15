"""
Mouse Controller Module
Controls mouse cursor and clicks using hand gestures
"""
import pyautogui
import time
import numpy as np


class MouseController:
    """Controls mouse with gestures"""
    
    def __init__(self, screen_width=1920, screen_height=1080, frame_width=640, frame_height=480):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.last_action_time = 0
        self.action_cooldown = 0.3
        
        # Smoothing
        self.smoothing = 5
        self.prev_x, self.prev_y = 0, 0
        
        # Scroll settings
        self.scroll_amount = 20
        
        # PyAutoGUI settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
        
        # Get actual screen size
        screen_size = pyautogui.size()
        self.screen_width = screen_size.width
        self.screen_height = screen_size.height
    
    def can_perform_action(self):
        """Check if enough time has passed since last action"""
        current_time = time.time()
        if current_time - self.last_action_time >= self.action_cooldown:
            self.last_action_time = current_time
            return True
        return False
    
    def move_cursor(self, x, y):
        """
        Move cursor based on hand position
        
        Args:
            x: X coordinate in frame
            y: Y coordinate in frame
        """
        # Convert frame coordinates to screen coordinates
        screen_x = np.interp(x, [0, self.frame_width], [0, self.screen_width])
        screen_y = np.interp(y, [0, self.frame_height], [0, self.screen_height])
        
        # Smoothing
        current_x = self.prev_x + (screen_x - self.prev_x) / self.smoothing
        current_y = self.prev_y + (screen_y - self.prev_y) / self.smoothing
        
        # Move mouse
        pyautogui.moveTo(current_x, current_y)
        
        self.prev_x, self.prev_y = current_x, current_y
    
    def click(self, button='left'):
        """Perform mouse click"""
        if not self.can_perform_action():
            return False
        
        pyautogui.click(button=button)
        return True
    
    def double_click(self):
        """Perform double click"""
        if not self.can_perform_action():
            return False
        
        pyautogui.doubleClick()
        return True
    
    def right_click(self):
        """Perform right click"""
        if not self.can_perform_action():
            return False
        
        pyautogui.rightClick()
        return True
    
    def drag_start(self):
        """Start dragging"""
        pyautogui.mouseDown()
    
    def drag_end(self):
        """End dragging"""
        pyautogui.mouseUp()
    
    def scroll_up(self, amount=None):
        """Scroll up"""
        if not self.can_perform_action():
            return False
        
        scroll = amount if amount else self.scroll_amount
        pyautogui.scroll(scroll)
        return True
    
    def scroll_down(self, amount=None):
        """Scroll down"""
        if not self.can_perform_action():
            return False
        
        scroll = amount if amount else self.scroll_amount
        pyautogui.scroll(-scroll)
        return True
    
    def handle_gesture(self, gesture, landmark_list=None):
        """
        Handle gesture and perform corresponding mouse action
        
        Args:
            gesture: Gesture name string
            landmark_list: Hand landmark positions (for cursor movement)
            
        Returns:
            Action description or None
        """
        # Move cursor with pointing gesture
        if gesture == 'point' and landmark_list and len(landmark_list) > 8:
            # Use index finger tip for cursor control
            x, y = landmark_list[8][1], landmark_list[8][2]
            self.move_cursor(x, y)
            return 'Moving Cursor'
        
        # Other gesture actions
        actions = {
            'pinch': lambda: self.click() and 'Click',
            'peace': lambda: self.right_click() and 'Right Click',
            'swipe_up': lambda: self.scroll_up() and 'Scroll Up',
            'swipe_down': lambda: self.scroll_down() and 'Scroll Down',
        }
        
        action = actions.get(gesture)
        if action:
            result = action()
            return result if result else None
        
        return None