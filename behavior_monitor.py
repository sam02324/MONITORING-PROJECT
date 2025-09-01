import cv2
import mediapipe as mp
import math

# --- You Can Adjust This Threshold ---
SMILE_THRESHOLD = 1.90
# ------------------------------------

# --- 1. INITIALIZE MEDIAPIPE MODELS ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate distance
def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

# --- 2. START WEBCAM AND PROCESS ---
cap = cv2.VideoCapture(0)
print("Starting combined behavior monitor. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- A. PROCESS FOR POSE (RAISED HAND) ---
    results_pose = pose.process(rgb_frame)
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Hand raised logic
        landmarks_pose = results_pose.pose_landmarks.landmark
        nose = landmarks_pose[mp_pose.PoseLandmark.NOSE]
        left_wrist = landmarks_pose[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks_pose[mp_pose.PoseLandmark.RIGHT_WRIST]

        if left_wrist.y < nose.y or right_wrist.y < nose.y:
            cv2.putText(frame, "HAND RAISED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- B. PROCESS FOR FACE MESH (SMILE) ---
    results_face = face_mesh.process(rgb_frame)
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            # NEW: Draw the face mesh to see what the AI sees
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())

            # Smile detection logic
            landmarks_face = face_landmarks.landmark
            p_mouth_left = landmarks_face[61]
            p_mouth_right = landmarks_face[291]
            p_eye_left = landmarks_face[33]
            p_eye_right = landmarks_face[133]

            mouth_width = distance(p_mouth_left, p_mouth_right)
            eye_width = distance(p_eye_left, p_eye_right)
            
            if eye_width > 0:
                smile_ratio = mouth_width / eye_width
                print(f"Smile Ratio: {smile_ratio:.2f}")
                
                if smile_ratio > SMILE_THRESHOLD:
                    cv2.putText(frame, "SMILING", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- C. DISPLAY THE FINAL FRAME ---
    cv2.imshow('Behavior Monitor', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()