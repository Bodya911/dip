import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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
    
    return scale_factor, ground_truth_faces, correctly_detected, missed_faces, false_positives

# Load Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Experiment settings
image_paths = ["C:/Users/bzh_p/univ/dip/lab2/ph1.jpg", "C:/Users/bzh_p/univ/dip/lab2/ph2.jpg"]  # Replace with your image paths
ground_truth_faces = [6, 5]  # Replace with actual number of faces in each image
scale_factors = np.arange(1.01, 3.01, 0.2)
min_neighbors = 5  # Fixed for experiment

# Store results
results = []
if not os.path.exists("C:/Users/bzh_p/univ/dip/lab2/results"):
    os.makedirs("C:/Users/bzh_p/univ/dip/lab2/results")

for img_path, gt_faces in zip(image_paths, ground_truth_faces):
    for sf in scale_factors:
        result = detect_faces(img_path, face_cascade, round(sf, 2), min_neighbors, gt_faces)
        results.append(result)

# Convert results to a structured table format
import pandas as pd
columns = ["ScaleFactor", "Truth faces", "Correctly Recognized", "Missed Faces", "False Positives"]
df = pd.DataFrame(results, columns=columns)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(df["ScaleFactor"], df["Correctly Recognized"], marker='o', label='Correctly Recognized')
plt.plot(df["ScaleFactor"], df["Missed Faces"], marker='s', label='Missed Faces')
plt.plot(df["ScaleFactor"], df["False Positives"], marker='^', label='False Positives')
plt.plot(df["ScaleFactor"], df["Truth faces"], marker='8', label='Truth faces')
plt.xlabel("Scale Factor")
plt.ylabel("Number of Faces")
plt.title("Effect of Scale Factor on Face Recognition")
plt.legend()
plt.grid()
plt.savefig("C:/Users/bzh_p/univ/dip/lab2/results/recognition_chart.png")
plt.show()

# Display results in table
print("Haar Cascade Experiment Results")
print(df)
df.to_csv("C:/Users/bzh_p/univ/dip/lab2/results/ex1_table.csv")