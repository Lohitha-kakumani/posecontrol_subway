import cv2
import mediapipe as mp
import pyautogui

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize pose and hands detection
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands()
draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
hands_joined = False

def get_key_action(pose_landmarks, hand_landmarks):
    global hands_joined

    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    nose = pose_landmarks[mp_pose.PoseLandmark.NOSE.value]

    center_x = (left_shoulder.x + right_shoulder.x) / 2

    # Move Left
    if center_x < 0.4:
        pyautogui.press('left')

    # Move Right
    elif center_x > 0.6:
        pyautogui.press('right')

    # If 2 hands detected
    if hand_landmarks and len(hand_landmarks) == 2:
        left_hand = hand_landmarks[0].landmark
        right_hand = hand_landmarks[1].landmark

        # Calculate hand distance for start gesture
        hands_distance = abs(left_hand[0].x - right_hand[0].x)
        if hands_distance < 0.05:
            if not hands_joined:
                pyautogui.press('space')  # Start game
                hands_joined = True
        else:
            hands_joined = False

        # Jump if both hands are above head (nose)
        if left_hand[0].y < nose.y and right_hand[0].y < nose.y:
            pyautogui.press('up')

        # Crouch if both hands are below hips
        if left_hand[0].y > left_hip.y and right_hand[0].y > right_hip.y:
            pyautogui.press('down')

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(img_rgb)
    hands_results = hands.process(img_rgb)

    if pose_results.pose_landmarks:
        draw.draw_landmarks(img, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        pose_landmarks = pose_results.pose_landmarks.landmark

        hand_landmarks = hands_results.multi_hand_landmarks if hands_results.multi_hand_landmarks else []
        if hand_landmarks:
            for hand in hand_landmarks:
                draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

        get_key_action(pose_landmarks, hand_landmarks)

    cv2.imshow("PoseControlled SubwaySurfer", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
