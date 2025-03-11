# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 12:35:26 2025

@author: bzh_p
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def detect_faces(image, face_cascade, scale_factor, min_neighbors):
    """Detect faces in a video frame using Haar Cascades."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    
    detected_faces = len(faces)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return detected_faces, image

# Load Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Experiment settings
best_combinations = [(1.1, 5), (1.2, 7), (1.3, 9)]  
video_source = 0  # Use default webcam

# Store results
results_video = []
if not os.path.exists("C:/Users/bzh_p/univ/dip/lab2/results/video"):
    os.makedirs("C:/Users/bzh_p/univ/dip/lab2/results/video")

# Open video camera
cap = cv2.VideoCapture(video_source)
frame_count = 0

while frame_count < 10:  # Capture 10 frames for testing
    ret, frame = cap.read()
    if not ret:
        break
    
    ground_truth_faces = 4
    
    for sf, mn in best_combinations:
        detected_faces, processed_frame = detect_faces(frame, face_cascade, sf, mn)
        correctly_detected = min(detected_faces, ground_truth_faces)
        missed_faces = max(0, ground_truth_faces - detected_faces)
        false_positives = max(0, detected_faces - ground_truth_faces)
        
        # Save frame image
        result_path = f"C:/Users/bzh_p/univ/dip/lab2/results/video/result_sf_{sf}_mn_{mn}_frame_{frame_count}.jpg"
        cv2.imwrite(result_path, processed_frame)
        
        results_video.append((sf, mn, ground_truth_faces, correctly_detected, missed_faces, false_positives))
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Convert results to structured table format
df_video = pd.DataFrame(results_video, columns=["ScaleFactor", "MinNeighbors", "Truth Faces", "Correctly Recognized", "Missed Faces", "False Positives"])

# Plot results for scaleFactor
plt.figure(figsize=(10, 6))
plt.plot(df_video["ScaleFactor"], df_video["Correctly Recognized"], marker='o', label='Correctly Recognized')
plt.plot(df_video["ScaleFactor"], df_video["Missed Faces"], marker='s', label='Missed Faces')
plt.plot(df_video["ScaleFactor"], df_video["False Positives"], marker='^', label='False Positives')
plt.plot(df_video["ScaleFactor"], df_video["Truth Faces"], marker='8', label='Truth Faces')
plt.xlabel("Scale Factor")
plt.ylabel("Number of Faces")
plt.title("Effect of Scale Factor on Face Recognition (Video)")
plt.legend()
plt.grid()
plt.savefig("C:/Users/bzh_p/univ/dip/lab2/results/video/recognition_chart_sf.png")
plt.show()

# Plot results for minNeighbors
plt.figure(figsize=(10, 6))
plt.plot(df_video["MinNeighbors"], df_video["Correctly Recognized"], marker='o', label='Correctly Recognized')
plt.plot(df_video["MinNeighbors"], df_video["Missed Faces"], marker='s', label='Missed Faces')
plt.plot(df_video["MinNeighbors"], df_video["False Positives"], marker='^', label='False Positives')
plt.plot(df_video["MinNeighbors"], df_video["Truth Faces"], marker='8', label='Truth Faces')
plt.xlabel("Min Neighbors")
plt.ylabel("Number of Faces")
plt.title("Effect of MinNeighbors on Face Recognition (Video)")
plt.legend()
plt.grid()
plt.savefig("C:/Users/bzh_p/univ/dip/lab2/results/video/recognition_chart_mn.png")
plt.show()

# Save and display results
df_video.to_csv("C:/Users/bzh_p/univ/dip/lab2/results/video/ex3_table.csv")
print("Haar Cascade Video Experiment Results")
print(df_video)
