# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 12:37:47 2025

@author: bzh_p
"""

import cv2
import os

# Set up Haar Cascade
face_cascade = cv2.CascadeClassifier("C:\\Users\\bzh_p\\univ\\dip\\lab3\\haarcascade_frontalface_default.xml")

# Update your best parameters from Lab 2 here
scaleFactor = 1.1
minNeighbors = 5

# Person info
person_name = "person2"  # change for each person
save_path = f"dataset/{person_name}/"

# Create directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        file_path = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(file_path, face)
        count += 1

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Recording Faces", frame)

    if cv2.waitKey(1) == 27:  # ESC to quit
        break

    if count >= 330:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} images to {save_path}")
