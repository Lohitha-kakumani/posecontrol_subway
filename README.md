🎮 Pose-Controlled Subway Surfer
This project allows users to control a game like Subway Surfer using body gestures captured via webcam, leveraging MediaPipe for real-time pose estimation and PyAutoGUI for simulating keyboard inputs.

✨ Features
Control character movement using your body:

Move left or right by shifting your shoulders.

Jump by raising your upper body.

Crouch by lowering your upper body.

Start the game or jump by joining both hands.

🧰 Technologies Used
Python

OpenCV

MediaPipe

PyAutoGUI

🚀 How to Run
Install dependencies:
pip install opencv-python mediapipe pyautogui
Run the script:
python main.py


🛑 Note
Do not move or delete the webcam or interrupt the pose detection during gameplay. Make sure your body is clearly visible to the webcam.
