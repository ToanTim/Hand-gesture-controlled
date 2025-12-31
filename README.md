# Hand Gesture Controlled Interface

A real-time hand gesture recognition system that enables touch-free control of computer applications using webcam-detected hand movements. Built with OpenCV, MediaPipe, and PyAutoGUI.

## Features

- 🖐️ **Real-time Hand Detection**: Uses MediaPipe for accurate hand landmark detection
- 👆 **Gesture Recognition**: Recognizes multiple hand gestures (fist, palm, pointing, peace sign, pinch, swipes)
- 📄 **PDF Control**: Navigate and control PDF viewers with intuitive gestures
- 🎵 **Media Control**: Control media playback, volume, and track navigation
- 🖱️ **Mouse Control**: Move cursor and perform clicks using hand movements
- 🔄 **Multiple Modes**: Switch between general, PDF, and media control modes
- 📊 **Visual Feedback**: Real-time display of detected gestures and actions

## Requirements

- Python 3.7+
- Webcam
- Operating System: Windows, macOS, or Linux

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ToanTim/Hand-gesture-controlled.git
cd Hand-gesture-controlled
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Application

Start the application with a specific mode:

```bash
# General mouse control mode (default)
python main.py

# PDF control mode
python main.py pdf

# Media control mode
python main.py media
```

### Using Example Scripts

Run pre-configured example scripts:

```bash
# Mouse control
python example_mouse_control.py

# PDF control (make sure a PDF viewer is open)
python example_pdf_control.py

# Media control (make sure a media player is open)
python example_media_control.py
```

### Controls

While the application is running:
- **'q'**: Quit the application
- **'m'**: Switch between modes (general → pdf → media → general)

## Gesture Guide

### General (Mouse Control) Mode

| Gesture | Action | Description |
|---------|--------|-------------|
| ☝️ Pointing | Move Cursor | Move your index finger to control cursor |
| 🤏 Pinch | Click | Bring thumb and index finger together |
| ✌️ Peace Sign | Right Click | Index and middle fingers up |
| 👆 Swipe Up | Scroll Up | Move hand upward |
| 👇 Swipe Down | Scroll Down | Move hand downward |

### PDF Control Mode

| Gesture | Action | Description |
|---------|--------|-------------|
| 👆 Swipe Up | Scroll Up | Scroll up in document |
| 👇 Swipe Down | Scroll Down | Scroll down in document |
| 👉 Swipe Right | Next Page | Go to next page |
| 👈 Swipe Left | Previous Page | Go to previous page |
| 🤏 Pinch | Zoom Out | Decrease zoom level |
| ✋ Palm Open | Reset Zoom | Reset to 100% zoom |

### Media Control Mode

| Gesture | Action | Description |
|---------|--------|-------------|
| ✋ Palm Open | Play/Pause | Toggle playback |
| 👉 Swipe Right | Next Track | Skip to next track |
| 👈 Swipe Left | Previous Track | Go to previous track |
| 👆 Swipe Up | Volume Up | Increase volume |
| 👇 Swipe Down | Volume Down | Decrease volume |
| ✊ Fist | Stop | Stop playback |
| ✌️ Peace Sign | Mute/Unmute | Toggle mute |

## How It Works

### Architecture

The system consists of several modules:

1. **hand_detector.py**: Uses MediaPipe to detect hand landmarks in real-time
2. **gesture_recognizer.py**: Analyzes hand positions to recognize specific gestures
3. **pdf_controller.py**: Translates gestures to PDF viewer controls
4. **media_controller.py**: Translates gestures to media player controls
5. **mouse_controller.py**: Translates gestures to mouse movements and clicks
6. **main.py**: Main application that integrates all components

### Technology Stack

- **OpenCV**: Video capture and frame processing
- **MediaPipe**: Hand landmark detection and tracking
- **PyAutoGUI**: System-level input automation
- **NumPy**: Numerical computations for gesture recognition

## Tips for Best Performance

1. **Lighting**: Ensure good lighting conditions for better hand detection
2. **Background**: Use a plain background for improved accuracy
3. **Distance**: Keep your hand 1-2 feet from the camera
4. **Camera Position**: Position camera at eye level or slightly above
5. **Calibration**: Give the system a moment to detect your hand initially
6. **Gesture Speed**: Perform gestures at moderate speed for better recognition

## Troubleshooting

### Webcam Not Detected
- Check if webcam is properly connected
- Verify webcam permissions in system settings
- Try changing the camera ID in `initialize_camera()` method

### Poor Gesture Recognition
- Improve lighting conditions
- Ensure hand is fully visible in frame
- Avoid cluttered backgrounds
- Adjust detection confidence in `HandDetector` initialization

### PyAutoGUI Not Working
- On Linux, you may need to install additional dependencies:
  ```bash
  sudo apt-get install python3-tk python3-dev
  ```
- On macOS, grant accessibility permissions to Terminal/Python

## Project Structure

```
Hand-gesture-controlled/
├── main.py                      # Main application
├── hand_detector.py             # Hand detection module
├── gesture_recognizer.py        # Gesture recognition module
├── pdf_controller.py            # PDF control module
├── media_controller.py          # Media control module
├── mouse_controller.py          # Mouse control module
├── example_pdf_control.py       # PDF control example
├── example_media_control.py     # Media control example
├── example_mouse_control.py     # Mouse control example
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MediaPipe by Google for hand tracking solution
- OpenCV community for computer vision tools
- PyAutoGUI for cross-platform GUI automation

## Future Enhancements

- [ ] Add more gesture types
- [ ] Implement gesture customization
- [ ] Add voice command integration
- [ ] Support for two-hand gestures
- [ ] Add gesture recording and replay
- [ ] Machine learning-based custom gesture training
- [ ] Mobile app companion

## Contact

For questions or suggestions, please open an issue on GitHub.