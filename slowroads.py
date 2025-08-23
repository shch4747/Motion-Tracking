import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import time
import math


# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Setup video capture
cap = cv2.VideoCapture(0)

# Confidence thresholds for detection and tracking
# Feel free to adjust these values
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

# Main logic to detect specific poses
with mp_pose.Pose(
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence) as pose:
    normal_left_hip_y=0
    normal_right_hip_y=0
    prev_left_ankle_y=0
    prev_right_ankle_y=0
    first_frame = False
    keyboard=Controller()
    left_hand=False
    right_hand=False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and get pose landmarks
        results = pose.process(rgb_frame)

        # Draw the pose landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Get landmark coordinates for easy access
            landmarks = results.pose_landmarks.landmark

            if not first_frame:
                normal_left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
                normal_right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
                first_frame = True
            
            # --- Pose Detection Logic ---
            steer_threshold = 0.05
            # Left Hand Raised Detection
            # A hand is raised if the wrist's y-coordinate is above the shoulder's y-coordinate
            # Y-coordinates decrease as you go up in an image
            left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
            left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
            left_steer_dist = abs(left_shoulder_x - left_wrist_x)
            if (left_steer_dist > steer_threshold):
                sleep_time = math.log(1.66*left_steer_dist + 0.917)
                keyboard.press("a")
                time.sleep(sleep_time)
                keyboard.release("a")
                cv2.putText(frame, str(sleep_time), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Right Hand Raised Detection
            right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            right_steer_dist = abs(right_wrist_y - right_shoulder_y)
            if (right_steer_dist > steer_threshold):
                sleep_time = math.log(1.66*right_steer_dist + 0.917)
                keyboard.press("d")
                time.sleep(sleep_time)
                keyboard.release("d")
                cv2.putText(frame, str(sleep_time), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Duck Pose Detection
            # A person is ducking if their hips are significantly lower than a standing position
            left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
            # We can use a simple threshold; adjust this value based on your camera and setup
            # Example: if hips are in the lower 50% of the frame
            frame_height, _, _ = frame.shape
            duck_threshold = 0.05 # Adjust as needed (30% of frame height)
            if (left_hip_y - normal_left_hip_y) > duck_threshold and (right_hip_y - normal_right_hip_y) > duck_threshold:
                # keyboard.press("s")
                # time.sleep(0.5)
                # keyboard.release("s")
                cv2.putText(frame, "DUCK", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Display the frame
        cv2.imshow('Pose Detection', frame)

        # Press 'q' to exit the video stream
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()