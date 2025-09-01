import cv2
import os
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import face_recognition
import mediapipe as mp
import math
import time

# --- 1. INITIALIZE MODELS AND SETTINGS ---
print("Loading models...")
# Face Recognition Models
known_face_encodings = []
known_face_names = []
KNOWN_FACES_DIR = "known_faces"

if os.path.exists(KNOWN_FACES_DIR):
    for filename in os.listdir(KNOWN_FACES_DIR):
        try:
            image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
            encoding = face_recognition.face_encodings(image)[0]
            name = os.path.splitext(filename)[0].replace("_", " ").title()
            known_face_encodings.append(encoding)
            known_face_names.append(name)
        except IndexError:
            pass
print("Known faces loaded.")

# YOLO Model & Anti-Spoofing Model
yolo_model = YOLO('yolov8s.pt') 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True)
print("All models loaded.")

# Liveness Tracking Variables
liveness_tracker = {}
LIVENESS_TIMEOUT = 10 

def distance(p1, p2, frame_shape):
    h, w, _ = frame_shape
    return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

# --- 2. START WEBCAM AND PROCESS ---
video_capture = cv2.VideoCapture(0)
print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_h, frame_w, _ = frame.shape

    # --- A. GATHER ALL FACE INFORMATION FIRST ---
    all_detected_faces = []
    
    # Run detections once
    all_face_landmarks = face_mesh.process(rgb_frame).multi_face_landmarks or []
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Recognize the person
        name = "Unknown"
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        # Check liveness for this specific face
        is_live_from_blink = False
        for face_landmarks in all_face_landmarks:
            center_x = int(np.mean([lm.x for lm in face_landmarks.landmark]) * frame_w)
            center_y = int(np.mean([lm.y for lm in face_landmarks.landmark]) * frame_h)
            if left < center_x < right and top < center_y < bottom:
                ear = 0.5 # Default EAR for open eye
                p_right_top, p_right_bottom = face_landmarks.landmark[159], face_landmarks.landmark[145]
                p_right_left, p_right_right = face_landmarks.landmark[33], face_landmarks.landmark[133]
                vertical_dist = distance(p_right_top, p_right_bottom, frame.shape)
                horizontal_dist = distance(p_right_left, p_right_right, frame.shape)
                if horizontal_dist > 0:
                    ear = vertical_dist / horizontal_dist
                if ear < 0.25:
                    is_live_from_blink = True
                    liveness_tracker[name] = time.time()
                break
        
        # Check overall liveness status from memory
        is_currently_live = False
        if name in liveness_tracker and time.time() - liveness_tracker[name] < LIVENESS_TIMEOUT:
            is_currently_live = True

        # Store all info for this face
        all_detected_faces.append({
            "name": name,
            "location": (top, right, bottom, left),
            "is_live": is_currently_live
        })

    # --- B. PRIORITIZE AND DRAW FACES (NO DUPLICATES) ---
    processed_names_this_frame = set()
    # Sort faces to prioritize live ones
    sorted_faces = sorted(all_detected_faces, key=lambda x: x['is_live'], reverse=True)

    for face in sorted_faces:
        name = face["name"]
        if name == "Unknown" or name not in processed_names_this_frame:
            (top, right, bottom, left) = face["location"]
            is_live = face["is_live"]
            
            liveness_status = "Live" if is_live else "Unverified"
            box_color = (0, 255, 0) if is_live else (0, 0, 255)
            
            label_text = f"{name} ({liveness_status})"
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
            cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            
            if name != "Unknown":
                processed_names_this_frame.add(name)


    # ... (YOLO and Timestamp code remains the same)

    cv2.imshow('Integrated Monitoring System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
