import cv2
import numpy as np

# Load trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_faces.yml")  # Load trained faces

# Load Haar Cascade (for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") 

# Employee names (same as in training)
employee_names = {0: "adhavan", 1: "basheer", 2: "harish", 3: "keshav", 4: "venkatesh"}

# Open webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]  # Crop face
        
        # Recognize the face
        label, confidence = recognizer.predict(face)

        if confidence < 200:  # Lower confidence = better match
            name = employee_names.get(label, "Unknown")  # Recognized employee
        else:
            name = "Customer"  # Unrecognized face

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()