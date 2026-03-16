import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import time
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# --- Tunable parameters ---
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5
STEER_THRESHOLD          = 0.05  # horizontal wrist-to-shoulder distance to trigger steering
DUCK_THRESHOLD           = 0.05  # hip drop fraction below calibrated baseline

# Steering key hold time scales logarithmically with wrist distance:
# hold_time = log(1.66 * dist + 0.917)
# At dist = STEER_THRESHOLD (0.05): ~0.01s  |  At dist = 0.5: ~0.48s

with mp_pose.Pose(
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose:

    normal_left_hip_y  = 0
    normal_right_hip_y = 0
    first_frame        = False
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
            if not first_frame:
                normal_left_hip_y  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
                normal_right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
                first_frame        = True

            left_wrist_x   = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
            left_shoulder_x  = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
            right_wrist_x  = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
            right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x

            # Steer left: left wrist moves away from left shoulder (A)
            left_dist = abs(left_shoulder_x - left_wrist_x)
            if left_dist > STEER_THRESHOLD:
                hold_time = math.log(1.66 * left_dist + 0.917)
                keyboard.press("a")
                time.sleep(hold_time)
                keyboard.release("a")
                cv2.putText(frame, f"LEFT ({hold_time:.2f}s)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Steer right: right wrist moves away from right shoulder (D)
            right_dist = abs(right_wrist_x - right_shoulder_x)
            if right_dist > STEER_THRESHOLD:
                hold_time = math.log(1.66 * right_dist + 0.917)
                keyboard.press("d")
                time.sleep(hold_time)
                keyboard.release("d")
                cv2.putText(frame, f"RIGHT ({hold_time:.2f}s)", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Duck: hips drop below baseline (S)
            left_hip_y  = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
            if (left_hip_y - normal_left_hip_y > DUCK_THRESHOLD and
                    right_hip_y - normal_right_hip_y > DUCK_THRESHOLD):
                cv2.putText(frame, "DUCK", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
