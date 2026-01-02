"""
PDF Controller Module
Controls PDF viewing applications using hand gestures
"""
import pyautogui
import time


class PDFController:
    """Controls PDF applications with gestures"""
    
    def __init__(self):
        self.last_action_time = 0
        self.action_cooldown = 0.5  # Seconds between actions
        self.scroll_amount = 100
        
        # Disable PyAutoGUI failsafe for smoother operation
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
    
    def can_perform_action(self):
        """Check if enough time has passed since last action"""
        current_time = time.time()
        if current_time - self.last_action_time >= self.action_cooldown:
            self.last_action_time = current_time
            return True
        return False
    
    def scroll_up(self, amount=None):
        """Scroll PDF up"""
        if not self.can_perform_action():
            return False
        
        scroll = amount if amount else self.scroll_amount
        pyautogui.scroll(scroll)
        return True
    
    def scroll_down(self, amount=None):
        """Scroll PDF down"""
        if not self.can_perform_action():
            return False
        
        scroll = amount if amount else self.scroll_amount
        pyautogui.scroll(-scroll)
        return True
    
    def next_page(self):
        """Go to next page"""
        if not self.can_perform_action():
            return False
        
        pyautogui.press('right')
        return True
    
    def previous_page(self):
        """Go to previous page"""
        if not self.can_perform_action():
            return False
        
        pyautogui.press('left')
        return True
    
    def zoom_in(self):
        """Zoom in on PDF"""
        if not self.can_perform_action():
            return False
        
        pyautogui.hotkey('ctrl', '+')
        return True
    
    def zoom_out(self):
        """Zoom out on PDF"""
        if not self.can_perform_action():
            return False
        
        pyautogui.hotkey('ctrl', '-')
        return True
    
    def reset_zoom(self):
        """Reset zoom to 100%"""
        if not self.can_perform_action():
            return False
        
        pyautogui.hotkey('ctrl', '0')
        return True
    
    def handle_gesture(self, gesture):
        """
        Handle gesture and perform corresponding PDF action
        
        Args:
            gesture: Gesture name string
            
        Returns:
            Action description or None
        """
        actions = {
            'point': lambda: self.scroll_up() and 'Scrolling Up',
            'peace': lambda: self.scroll_down() and 'Scrolling Down',
            'swipe_right': lambda: self.next_page() and 'Next Page',
            'swipe_left': lambda: self.previous_page() and 'Previous Page',
            'pinch': lambda: self.zoom_out() and 'Zoom Out',
            'open_pinch': lambda: self.zoom_in() and 'Zoom In',
            'palm': lambda: self.reset_zoom() and 'Reset Zoom',
            'fist': lambda: None,  # No action for fist in PDF mode
        }
        
        action = actions.get(gesture)
        if action:
            result = action()
            return result if result else None
        
        return None
