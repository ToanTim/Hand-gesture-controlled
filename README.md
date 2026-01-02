# Hand Gesture Controlled Interface

Control your computer using hand gestures! This project uses MediaPipe and OpenCV to detect hand gestures and control various applications including PDF viewers, media players, and mouse movements.

## 🎥 Demo Videos

### PDF Navigation

[![PDF Control Demo]](https://youtu.be/MmyUTw7ZpCQ)

## ✨ Features

- **Hand Detection**: Real-time hand tracking using MediaPipe
- **Gesture Recognition**: Recognizes 10+ different gestures
- **Multiple Control Modes**:
  - 📄 **PDF Mode**: Navigate and zoom PDF documents

## 🎮 Supported Gestures

| Gesture        | PDF Mode      |
| -------------- | ------------- |
| 👆 Point       | Scroll Up     |
| ✌️ Peace       | Scroll Down   |
| 👌 Pinch       | Zoom Out      |
| 🖐️ Palm        | Reset Zoom    |
| ✊ Fist        | Stop          |
| 👍 Thumbs Up   | -             |
| ⬅️ Swipe Left  | Previous Page |
| ➡️ Swipe Right | Next Page     |
| ⬆️ Swipe Up    | -             |
| ⬇️ Swipe Down  | -             |

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/ToanTim/Hand-gesture-controlled.git
cd Hand-gesture-controlled
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## 📖 Usage

Run the application with default mode (general):

```bash
python src/main.py
```

Or specify a mode:

```bash

# PDF control mode
python main.py pdf

```

### Controls

- **Q**: Quit application
- **M**: Switch between modes (General → PDF → Media)

## 📁 Project Structure

```
Hand-gesture-controlled/
├── main.py                      # Main application
├── hand_detector.py             # Hand detection module
├── gesture_recognizer.py        # Gesture recognition module
├── pdf_controller.py            # PDF control module
├── mouse_controller.py          # Mouse control module
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🛠️ Requirements

- Python 3.7+
- Webcam
- Dependencies:
  - OpenCV
  - MediaPipe
  - PyAutoGUI
  - NumPy

## 📝 Configuration

Edit `config/config.json` to customize:

- Gesture sensitivity
- Action cooldown periods
- Screen resolution
- Camera settings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [OpenCV](https://opencv.org/) for computer vision
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for system control

## 📧 Contact

- GitHub: [@ToanTim](https://github.com/ToanTim)
- Project Link: [https://github.com/ToanTim/Hand-gesture-controlled](https://github.com/ToanTim/Hand-gesture-controlled)

---

⭐ Star this repo if you find it helpful!
