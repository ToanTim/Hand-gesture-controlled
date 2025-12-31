"""
Media Player Controller Module
Controls media player applications using hand gestures
"""
import pyautogui
import time


class MediaController:
    """Controls media players with gestures"""
    
    def __init__(self):
        self.last_action_time = 0
        self.action_cooldown = 0.5  # Seconds between actions
        self.volume_step = 2
        
        # PyAutoGUI settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
    
    def can_perform_action(self):
        """Check if enough time has passed since last action"""
        current_time = time.time()
        if current_time - self.last_action_time >= self.action_cooldown:
            self.last_action_time = current_time
            return True
        return False
    
    def play_pause(self):
        """Toggle play/pause"""
        if not self.can_perform_action():
            return False
        
        pyautogui.press('space')
        return True
    
    def next_track(self):
        """Skip to next track"""
        if not self.can_perform_action():
            return False
        
        pyautogui.press('nexttrack')
        return True
    
    def previous_track(self):
        """Go to previous track"""
        if not self.can_perform_action():
            return False
        
        pyautogui.press('prevtrack')
        return True
    
    def volume_up(self):
        """Increase volume"""
        if not self.can_perform_action():
            return False
        
        for _ in range(self.volume_step):
            pyautogui.press('volumeup')
        return True
    
    def volume_down(self):
        """Decrease volume"""
        if not self.can_perform_action():
            return False
        
        for _ in range(self.volume_step):
            pyautogui.press('volumedown')
        return True
    
    def mute(self):
        """Toggle mute"""
        if not self.can_perform_action():
            return False
        
        pyautogui.press('volumemute')
        return True
    
    def stop(self):
        """Stop playback"""
        if not self.can_perform_action():
            return False
        
        pyautogui.press('stop')
        return True
    
    def seek_forward(self):
        """Seek forward"""
        if not self.can_perform_action():
            return False
        
        pyautogui.press('right')
        return True
    
    def seek_backward(self):
        """Seek backward"""
        if not self.can_perform_action():
            return False
        
        pyautogui.press('left')
        return True
    
    def handle_gesture(self, gesture):
        """
        Handle gesture and perform corresponding media action
        
        Args:
            gesture: Gesture name string
            
        Returns:
            Action description or None
        """
        actions = {
            'palm': lambda: self.play_pause() and 'Play/Pause',
            'swipe_right': lambda: self.next_track() and 'Next Track',
            'swipe_left': lambda: self.previous_track() and 'Previous Track',
            'swipe_up': lambda: self.volume_up() and 'Volume Up',
            'swipe_down': lambda: self.volume_down() and 'Volume Down',
            'fist': lambda: self.stop() and 'Stop',
            'peace': lambda: self.mute() and 'Mute/Unmute',
        }
        
        action = actions.get(gesture)
        if action:
            result = action()
            return result if result else None
        
        return None
