import cv2
import mediapipe as mp
import time

# --- 1. INITIALIZE MODELS AND SETTINGS ---
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- 2. SETUP FOR MALPRACTICE DETECTION ---
# Variables to track head turning over time
head_turned_start_time = None
TURN_DURATION_ALERT = 3.0  # Alert if head is turned for 3 seconds

# --- 3. START WEBCAM AND PROCESS ---
cap = cv2.VideoCapture(0)
print("Starting Exam Proctor. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame for pose
    results = pose.process(rgb_frame)

    alert = False
    direction = "Forward"

    if results.pose_landmarks:
        # Draw the skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # --- Head Pose Estimation Logic ---
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates for shoulders and nose
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        
        # Make sure landmarks are visible
        if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and nose.visibility > 0.5:
            # Calculate the center of the shoulders
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            
            # Check if the nose has moved significantly from the shoulder center
            # This indicates the head is turned relative to the torso
            turn_threshold = 0.08 # Adjust this value based on your camera setup
            
            if nose.x < shoulder_center_x - turn_threshold:
                direction = "Turned Left"
            elif nose.x > shoulder_center_x + turn_threshold:
                direction = "Turned Right"
            else:
                direction = "Forward"
        
        # --- Malpractice Timer Logic ---
        if direction != "Forward":
            if head_turned_start_time is None:
                # Start the timer if the head has just been turned
                head_turned_start_time = time.time()
            else:
                # If head is still turned, check how long it's been
                elapsed_time = time.time() - head_turned_start_time
                if elapsed_time > TURN_DURATION_ALERT:
                    alert = True
        else:
            # Reset the timer if the head is facing forward
            head_turned_start_time = None

    # --- DRAWING ---
    # Display the current head direction
    cv2.putText(frame, f"Direction: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # If an alert is triggered, display it prominently
    if alert:
        alert_text = "CHEATING ALERT!"
        (text_width, text_height), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)
        cv2.rectangle(frame, (frame_w // 2 - text_width // 2 - 10, frame_h // 2 - text_height // 2 - 20),
                      (frame_w // 2 + text_width // 2 + 10, frame_h // 2 + text_height // 2 + 10), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, alert_text, (frame_w // 2 - text_width // 2, frame_h // 2 + text_height // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
    
    cv2.imshow('Exam Proctor', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()