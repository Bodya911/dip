# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 12:58:14 2025

@author: bzh_p
"""

import cv2
import numpy as np
import os
import statistics
import matplotlib.pyplot as plt

# Load dataset function (same as before)
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

# --- EXPERIMENT STARTS HERE ---

# Haar cascade
face_cascade = cv2.CascadeClassifier('C:\\Users\\bzh_p\\univ\\dip\\lab3\\haarcascade_frontalface_default.xml')
scaleFactor = 1.1
minNeighbors = 5

dataset_path = "dataset/"
images, labels, label_map = load_dataset(dataset_path)

# Test settings
num_components_list = [80, 100, 120, 140, 160]  # You can change/add more
threshold = 100000

# Store results
experiment_results = []

for num_components in num_components_list:
    print(f"\nTesting num_components = {num_components}")
    model = cv2.face.EigenFaceRecognizer_create(num_components=num_components)
    model.train(images, labels)

    # Start webcam
    cap = cv2.VideoCapture(0)

    correct = 0
    false = 0
    unrecognized = 0
    confidence_values = []

    test_samples = 30  # Number of test attempts to record per num_components
    samples_taken = 0

    while samples_taken < test_samples:
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
            confidence_values.append(confidence)

            print(f"Predicted: {label_map.get(label, 'Unknown')} with confidence {confidence:.2f}")

            if confidence < threshold:
                # Recognized - you can add more manual checking here
                correct += 1
            else:
                unrecognized += 1

            samples_taken += 1

            # Save successful examples
            if samples_taken == 5:  # Save 5th recognition image as an example
                filename = f"successful_numcomp{num_components}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved example image: {filename}")

            break  # Only one face per frame

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Statistics
    min_conf = np.min(confidence_values)
    max_conf = np.max(confidence_values)
    mean_conf = np.mean(confidence_values)
    std_conf = np.std(confidence_values)

    # Save result
    experiment_results.append({
        "num_components": num_components,
        "correct": correct,
        "false": false,
        "unrecognized": unrecognized,
        "min_confidence": min_conf,
        "max_confidence": max_conf,
        "mean_confidence": mean_conf,
        "std_confidence": std_conf
    })

# Print results in table format
print("\n--- Experiment Results ---")
print(f"{'num_components':<15} {'Correct':<10} {'False':<10} {'Unrecognized':<15} {'Min Conf':<10} {'Max Conf':<10} {'Mean Conf':<12} {'Std Conf'}")
for result in experiment_results:
    print(f"{result['num_components']:<15} {result['correct']:<10} {result['false']:<10} {result['unrecognized']:<15} {result['min_confidence']:<10.2f} {result['max_confidence']:<10.2f} {result['mean_confidence']:<12.2f} {result['std_confidence']:.2f}")

# Plot chart
x = [r['num_components'] for r in experiment_results]
y = [r['mean_confidence'] for r in experiment_results]

plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o')
plt.title('Effect of num_components on Mean Confidence (Eigenfaces)')
plt.xlabel('Number of Components')
plt.ylabel('Mean Confidence')
plt.grid(True)
plt.savefig('eigenfaces_confidence_plot.png')
plt.show()
