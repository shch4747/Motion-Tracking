import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
import time


# Initialize mediapipe pose, since we'll be looking at the pose of the user to select gestures.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Setup video capture
cap = cv2.VideoCapture(0)

# Confidence thresholds for detection and tracking
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

# Main logic to detect specific poses
with mp_pose.Pose(
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence) as pose:

    #Keyboard to simulate key presses
    keyboard=Controller()

    #Thresholds (Values are in (% frame distance)/100)
    side_threshold = 0.13 #->How much you have to move your hand horizontally to move once in that direction
    duck_threshold = 0.02 #->How much you have to duck 
    side_overflow_threshold = 0.02 #->How much below your shoulder your arms need to be during sidewise movement
    jump_threshold = 0.1 #->How much your arms need to be above your shoulders to detect jump

    #Various parameters used
    normal_left_hip_y=0
    normal_right_hip_y=0
    reset_frame = True #Keeps track of recalibrating frames
    left_hand=False #True when left hand is moved
    right_hand=False #True when right hand is moved
    

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

            if reset_frame:
                normal_left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
                normal_right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
                reset_frame = False
            
            # --- Pose Detection Logic ---

            #LEFT AND RIGHT HAND DETECTION

            left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_wrist_x = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x
            right_wrist_x = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x
            left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
            right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x

            if (left_wrist_x - left_shoulder_x > side_threshold) and (not left_hand) and (left_wrist_y - left_shoulder_y > side_overflow_threshold):
                left_hand=True
                keyboard.press("a")
                time.sleep(0.01)
                keyboard.release("a")
                cv2.putText(frame, "LEFT HAND", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
            if (left_wrist_x-left_shoulder_x<side_threshold) and (left_hand):
                left_hand=False

            # Right Hand Raised Detection
            if (right_shoulder_x - right_wrist_x > side_threshold) and (not right_hand) and (right_wrist_y - right_shoulder_y > side_overflow_threshold):
                right_hand=True
                keyboard.press("d")
                time.sleep(0.01)
                keyboard.release("d")
                cv2.putText(frame, "RIGHT HAND", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if (right_shoulder_x - right_wrist_x < side_threshold) and right_hand:
                right_hand=False

            #JUMP Detection

            if ((left_shoulder_y - left_wrist_y > jump_threshold) and (left_wrist_x-left_shoulder_x<side_threshold)) or ((right_shoulder_y - right_wrist_y > jump_threshold) and (right_shoulder_x - right_wrist_x < side_threshold)):
                keyboard.press("w")
                time.sleep(0.01)
                keyboard.release("w")
                left_hand=True
                right_hand=True
                cv2.putText(frame,"JUMP", (30,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

            #RESET Detection
            #You can reset by forming and X with you arms
            #Resetting essentially means recalibrating the waist height of the person
            #This should be done whenever the game shows movement issues and whenever the person playing changes

            if (right_wrist_x-left_wrist_x > 0.05):
                reset_frame = True
                cv2.putText(frame, "RESET", (30, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # DUCK Detection

            left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
            if (left_hip_y - normal_left_hip_y) > duck_threshold and (right_hip_y - normal_right_hip_y) > duck_threshold:
                keyboard.press("s")
                time.sleep(0.1)
                keyboard.release("s")
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