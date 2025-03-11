# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 12:23:02 2025

@author: bzh_p
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def detect_faces(image_path, face_cascade, scale_factor, min_neighbors, ground_truth_faces):
    """Detect faces in an image using Haar Cascades."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    
    detected_faces = len(faces)
    correctly_detected = min(detected_faces, ground_truth_faces)
    missed_faces = max(0, ground_truth_faces - detected_faces)
    false_positives = max(0, detected_faces - ground_truth_faces)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Save and display image
    result_path = f"C:/Users/bzh_p/univ/dip/lab2/results/result_sf_{scale_factor}_mn_{min_neighbors}.jpg"
    cv2.imwrite(result_path, img)
    cv2.imshow(f"ScaleFactor: {scale_factor}, MinNeighbors: {min_neighbors}", img)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    return scale_factor, min_neighbors, ground_truth_faces, correctly_detected, missed_faces, false_positives

# Load Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Experiment settings
image_paths = ["C:/Users/bzh_p/univ/dip/lab2/ph1.jpg", "C:/Users/bzh_p/univ/dip/lab2/ph2.jpg"]  # Replace with your image paths
ground_truth_faces = [6, 5]  # Replace with actual number of faces in each image
scale_factors = np.arange(1.01, 3.01, 0.2)
min_neighbors_values = range(1, 11, 2)  # Testing minNeighbors values from 1 to 10 with steps

# Store results
results_sf = []
results_mn = []
if not os.path.exists("C:/Users/bzh_p/univ/dip/lab2/results"):
    os.makedirs("C:/Users/bzh_p/univ/dip/lab2/results")

# First experiment: Varying scaleFactor with fixed minNeighbors
fixed_min_neighbors = 5  # Fixed for first experiment
for img_path, gt_faces in zip(image_paths, ground_truth_faces):
    for sf in scale_factors:
        result = detect_faces(img_path, face_cascade, round(sf, 2), fixed_min_neighbors, gt_faces)
        results_sf.append(result)

# Find best scaleFactor with highest correct detections
df_sf = pd.DataFrame(results_sf, columns=["ScaleFactor", "MinNeighbors", "Truth Faces", "Correctly Recognized", "Missed Faces", "False Positives"])
best_scale_factor = df_sf.loc[df_sf["Correctly Recognized"].idxmax(), "ScaleFactor"]

# Second experiment: Varying minNeighbors with best scaleFactor
for img_path, gt_faces in zip(image_paths, ground_truth_faces):
    for mn in min_neighbors_values:
        result = detect_faces(img_path, face_cascade, best_scale_factor, mn, gt_faces)
        results_mn.append(result)

# Convert results to structured table format
df_mn = pd.DataFrame(results_mn, columns=["ScaleFactor", "MinNeighbors", "Truth Faces", "Correctly Recognized", "Missed Faces", "False Positives"])

# Plot results for minNeighbors
plt.figure(figsize=(10, 6))
plt.plot(df_mn["MinNeighbors"], df_mn["Correctly Recognized"], marker='o', label='Correctly Recognized')
plt.plot(df_mn["MinNeighbors"], df_mn["Missed Faces"], marker='s', label='Missed Faces')
plt.plot(df_mn["MinNeighbors"], df_mn["False Positives"], marker='^', label='False Positives')
plt.plot(df_mn["MinNeighbors"], df_mn["Truth Faces"], marker='8', label='Truth Faces')
plt.xlabel("Min Neighbors")
plt.ylabel("Number of Faces")
plt.title("Effect of MinNeighbors on Face Recognition")
plt.legend()
plt.grid()
plt.savefig("C:/Users/bzh_p/univ/dip/lab2/results/recognition_chart_mn.png")
plt.show()

# Save and display results
df_mn.to_csv("C:/Users/bzh_p/univ/dip/lab2/results/ex2_table.csv")
print("Haar Cascade MinNeighbors Experiment Results")
print(df_mn)
