#!/usr/bin/env python3
"""
PDF Control Example
Run this script to control PDF viewer with hand gestures
"""
from main import HandGestureApp

if __name__ == "__main__":
    print("=" * 50)
    print("PDF CONTROL MODE")
    print("=" * 50)
    print("\nGestures:")
    print("  👆 Swipe Up     - Scroll Up")
    print("  👇 Swipe Down   - Scroll Down")
    print("  👉 Swipe Right  - Next Page")
    print("  👈 Swipe Left   - Previous Page")
    print("  🤏 Pinch        - Zoom Out")
    print("  ✋ Palm Open    - Reset Zoom")
    print("\nMake sure your PDF viewer is open and active!")
    print("\nPress 'q' to quit\n")
    
    app = HandGestureApp(mode='pdf')
    app.run()
