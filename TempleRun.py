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
SIDE_THRESHOLD           = 0.13  # horizontal wrist-to-shoulder distance to trigger left/right
SIDE_OVERFLOW_THRESHOLD  = 0.02  # wrist must be this far below shoulder to avoid jump conflicts
DUCK_THRESHOLD           = 0.02  # hip drop fraction below calibrated baseline
JUMP_THRESHOLD           = 0.10  # wrist must be this far above shoulder to trigger jump

with mp_pose.Pose(
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose:

    normal_left_hip_y  = 0
    normal_right_hip_y = 0
    reset_frame        = True
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

            # Calibrate hip baseline on reset
            if reset_frame:
                normal_left_hip_y  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
                normal_right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
                reset_frame        = False

            left_wrist_y    = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            left_shoulder_y   = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_wrist_y   = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            right_shoulder_y  = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_wrist_x    = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
            right_wrist_x   = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
            left_shoulder_x   = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
            right_shoulder_x  = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x

            # Move left: left wrist extends past shoulder horizontally (A)
            if (left_wrist_x - left_shoulder_x > SIDE_THRESHOLD and
                    not left_hand and
                    left_wrist_y - left_shoulder_y > SIDE_OVERFLOW_THRESHOLD):
                left_hand = True
                keyboard.press("a")
                time.sleep(0.01)
                keyboard.release("a")
                cv2.putText(frame, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if left_wrist_x - left_shoulder_x < SIDE_THRESHOLD:
                left_hand = False

            # Move right: right wrist extends past shoulder horizontally (D)
            if (right_shoulder_x - right_wrist_x > SIDE_THRESHOLD and
                    not right_hand and
                    right_wrist_y - right_shoulder_y > SIDE_OVERFLOW_THRESHOLD):
                right_hand = True
                keyboard.press("d")
                time.sleep(0.01)
                keyboard.release("d")
                cv2.putText(frame, "RIGHT", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if right_shoulder_x - right_wrist_x < SIDE_THRESHOLD:
                right_hand = False

            # Jump: either wrist raised above shoulder while arm not extended sideways (W)
            left_jump  = (left_shoulder_y - left_wrist_y > JUMP_THRESHOLD and
                          left_wrist_x - left_shoulder_x < SIDE_THRESHOLD)
            right_jump = (right_shoulder_y - right_wrist_y > JUMP_THRESHOLD and
                          right_shoulder_x - right_wrist_x < SIDE_THRESHOLD)
            if left_jump or right_jump:
                keyboard.press("w")
                time.sleep(0.01)
                keyboard.release("w")
                left_hand  = True
                right_hand = True
                cv2.putText(frame, "JUMP", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Reset calibration: arms crossed (X shape)
            if right_wrist_x - left_wrist_x > 0.05:
                reset_frame = True
                cv2.putText(frame, "RESET", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Duck: hips drop below baseline (S)
            left_hip_y  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
            if (left_hip_y - normal_left_hip_y > DUCK_THRESHOLD and
                    right_hip_y - normal_right_hip_y > DUCK_THRESHOLD):
                keyboard.press("s")
                time.sleep(0.1)
                keyboard.release("s")
                cv2.putText(frame, "DUCK", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
