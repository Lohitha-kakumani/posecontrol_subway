import cv2
import mediapipe as mp
import pyautogui

# Initialize
cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
prev_position = None
hands_joined = False

def get_key_action(landmarks):
    global hands_joined

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    # Get hand distance
    hands_distance = abs(left_wrist.x - right_wrist.x)

    # Start game gesture (join hands)
    if hands_distance < 0.05:
        if not hands_joined:
            pyautogui.press('space')  # Start the game or jump
            hands_joined = True
    else:
        hands_joined = False

    center_x = (left_shoulder.x + right_shoulder.x) / 2

    # Move Left
    if center_x < 0.4:
        pyautogui.press('left')

    # Move Right
    elif center_x > 0.6:
        pyautogui.press('right')

    # Jump
    if left_shoulder.y - left_hip.y < 0.2:
        pyautogui.press('up')

    # Crouch
    elif left_shoulder.y - left_hip.y > 0.5:
        pyautogui.press('down')

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        get_key_action(landmarks)

    cv2.imshow("PoseControlled SubwaySurfer", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
