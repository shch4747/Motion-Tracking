import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# --- Tunable parameters ---
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5
DUCK_THRESHOLD           = 0.05  # hip drop fraction below calibrated baseline
JUMP_THRESHOLD           = 0.01  # ankle rise fraction per frame

with mp_pose.Pose(
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose:

    normal_left_hip_y  = 0
    normal_right_hip_y = 0
    prev_left_ankle_y  = 0
    prev_right_ankle_y = 0
    first_frame        = True
    left_hand          = False
    right_hand         = False
    keyboard           = Controller()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            landmarks = results.pose_landmarks.landmark

            # Calibrate hip baseline on first frame
            if first_frame:
                normal_left_hip_y  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
                normal_right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
                first_frame        = False

            left_wrist_y   = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            left_shoulder_y  = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_wrist_y  = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_wrist_x   = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
            right_wrist_x  = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x

            # Reset calibration: wrists crossed
            if right_wrist_x - left_wrist_x > 0.05:
                first_frame = True
                cv2.putText(frame, "RESET", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Both hands raised: jump (Space)
            if left_wrist_y < left_shoulder_y and right_wrist_y < right_shoulder_y:
                keyboard.press(Key.space)
                time.sleep(0.01)
                keyboard.release(Key.space)
                left_hand  = True
                right_hand = True
                cv2.putText(frame, "JUMP", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Left hand raised: move left (A)
            if left_wrist_y < left_shoulder_y and not left_hand:
                left_hand = True
                keyboard.press("a")
                time.sleep(0.1)
                keyboard.release("a")
                cv2.putText(frame, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if left_wrist_y > left_shoulder_y:
                left_hand = False

            # Right hand raised: move right (D)
            if right_wrist_y < right_shoulder_y and not right_hand:
                right_hand = True
                keyboard.press("d")
                time.sleep(0.1)
                keyboard.release("d")
                cv2.putText(frame, "RIGHT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if right_wrist_y > right_shoulder_y:
                right_hand = False

            # Duck: hips drop below baseline (S)
            left_hip_y  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
            if (left_hip_y - normal_left_hip_y > DUCK_THRESHOLD and
                    right_hip_y - normal_right_hip_y > DUCK_THRESHOLD):
                keyboard.press("s")
                time.sleep(0.1)
                keyboard.release("s")
                cv2.putText(frame, "DUCK", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Jump: ankles rise relative to previous frame (W)
            left_ankle_y  = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
            right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y
            if (prev_left_ankle_y - left_ankle_y > JUMP_THRESHOLD and
                    prev_right_ankle_y - right_ankle_y > JUMP_THRESHOLD):
                keyboard.press("w")
                time.sleep(0.1)
                keyboard.release("w")
                cv2.putText(frame, "JUMP", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            prev_left_ankle_y  = left_ankle_y
            prev_right_ankle_y = right_ankle_y

        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
