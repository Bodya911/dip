# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 13:27:46 2025

@author: bzh_p
"""

import cv2
import numpy as np
import os
import time

# Load dataset function
def load_dataset(dataset_path):
    images = []
    labels = []
    label_map = {}

    label_counter = 0
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        label_map[label_counter] = person_name

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(label_counter)

        label_counter += 1

    return images, np.array(labels), label_map

# Haar Cascade settings
face_cascade = cv2.CascadeClassifier('C:\\Users\\bzh_p\\univ\\dip\\lab3\\haarcascade_frontalface_default.xml')
scaleFactor = 1.1
minNeighbors = 5

# Load dataset
dataset_path = "dataset/"
images, labels, label_map = load_dataset(dataset_path)

# Best parameters (replace with your real values after experiments)
BEST_EIGEN_COMPONENTS = 100
BEST_EIGEN_THRESHOLD = 25000

BEST_FISHER_COMPONENTS = 1
BEST_FISHER_THRESHOLD = 2000

BEST_LBPH_RADIUS = 1
BEST_LBPH_GRID = 8
BEST_LBPH_THRESHOLD = 50

# Methods to test
methods = ["Eigenfaces", "Fisherfaces", "LBPH"]

# Function to create model based on method
def create_model(method):
    if method == "Eigenfaces":
        model = cv2.face.EigenFaceRecognizer_create(num_components=BEST_EIGEN_COMPONENTS)
    elif method == "Fisherfaces":
        model = cv2.face.FisherFaceRecognizer_create(num_components=BEST_FISHER_COMPONENTS)
    elif method == "LBPH":
        model = cv2.face.LBPHFaceRecognizer_create(radius=BEST_LBPH_RADIUS, neighbors=8, grid_x=BEST_LBPH_GRID, grid_y=BEST_LBPH_GRID)
    else:
        raise ValueError("Invalid method!")
    return model

# Function to get correct threshold based on method
def get_threshold(method):
    if method == "Eigenfaces":
        return BEST_EIGEN_THRESHOLD
    elif method == "Fisherfaces":
        return BEST_FISHER_THRESHOLD
    elif method == "LBPH":
        return BEST_LBPH_THRESHOLD

# --- Start testing ---
for method in methods:
    print(f"\n[INFO] Testing method: {method}")

    model = create_model(method)
    model.train(images, labels)
    threshold = get_threshold(method)

    cap = cv2.VideoCapture(0)
    frame_count = 0

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

            label, confidence = model.predict(face)

            if confidence < threshold:
                name = label_map.get(label, 'Unknown')
                color = (0, 255, 0)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                name = "Unknown"
                color = (0, 0, 255)
                cv2.putText(frame, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.imshow(f"{method} - Multi-face Recognition", frame)

        frame_count += 1

        # Save frame after 100 frames (~3-5 seconds)
        if frame_count == 100:
            filename = f"{method}_multi_faces_success.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved image: {filename}")

        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    time.sleep(2)  # Short pause before next method
