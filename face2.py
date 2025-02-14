import cv2
import numpy as np
import os

# Initialize OpenCV's face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load OpenCV's Haar Cascade (for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Path to employees folder
employee_folder = "known_faces"
employee_images = []
employee_labels = []
employee_names = {}

# Assign unique ID numbers to each employee
id_counter = 0

for file in os.listdir(employee_folder):
    if file.endswith(".jpg") or file.endswith(".png"):
        img_path = os.path.join(employee_folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Detect face in the image
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]  # Crop face
            employee_images.append(face)
            employee_labels.append(id_counter)
        
        employee_names[id_counter] = os.path.splitext(file)[0]  # Store name
        id_counter += 1

# Train the recognizer
recognizer.train(employee_images, np.array(employee_labels))

print("Training complete. Employees loaded:", employee_names)