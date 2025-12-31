#!/usr/bin/env python3
"""
Media Control Example
Run this script to control media player with hand gestures
"""
from main import HandGestureApp

if __name__ == "__main__":
    print("=" * 50)
    print("MEDIA CONTROL MODE")
    print("=" * 50)
    print("\nGestures:")
    print("  ✋ Palm Open    - Play/Pause")
    print("  👉 Swipe Right  - Next Track")
    print("  👈 Swipe Left   - Previous Track")
    print("  👆 Swipe Up     - Volume Up")
    print("  👇 Swipe Down   - Volume Down")
    print("  ✊ Fist         - Stop")
    print("  ✌️  Peace Sign   - Mute/Unmute")
    print("\nMake sure your media player is open and active!")
    print("\nPress 'q' to quit\n")
    
    app = HandGestureApp(mode='media')
    app.run()
