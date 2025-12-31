#!/usr/bin/env python3
"""
Mouse Control Example
Run this script to control mouse cursor with hand gestures
"""
from main import HandGestureApp

if __name__ == "__main__":
    print("=" * 50)
    print("MOUSE CONTROL MODE")
    print("=" * 50)
    print("\nGestures:")
    print("  ☝️  Point        - Move Cursor")
    print("  🤏 Pinch        - Click")
    print("  ✌️  Peace Sign   - Right Click")
    print("  👆 Swipe Up     - Scroll Up")
    print("  👇 Swipe Down   - Scroll Down")
    print("\nNote: Move your index finger to control the cursor")
    print("\nPress 'q' to quit\n")
    
    app = HandGestureApp(mode='general')
    app.run()
