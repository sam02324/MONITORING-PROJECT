import dlib
import face_recognition

print("? dlib is working, version:", dlib.__version__)

# Load the image
image = face_recognition.load_image_file("test.jpg")

# Detect faces
faces = face_recognition.face_locations(image)

print("? Found", len(faces), "face(s) in the image")
