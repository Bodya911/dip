# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:02:05 2025

@author: bzh_p
"""

import cv2
from brisque import BRISQUE
import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to capture an image from the webcam
def capture_image(filename='original_image.png'):
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
    cap.release()
    cv2.destroyAllWindows()
    return filename

# Function to calculate image quality, file size, and image resolution
def analyze_image(filepath):
    obj = BRISQUE(url=False)
    quality_score = obj.score(cv2.imread(filepath))
    file_size = os.path.getsize(filepath) / 1024  # Convert to KB
    image = cv2.imread(filepath)
    resolution = f"{image.shape[1]}x{image.shape[0]}"  # Width x Height
    return quality_score, file_size, resolution

# Function to save images in different formats
def save_in_formats(image_path, output_prefix):
    image = cv2.imread(image_path)
    formats = ['bmp', 'jpg', 'png']
    saved_files = []
    for fmt in formats:
        filename = f"{output_prefix}.{fmt}"
        cv2.imwrite(filename, image)
        saved_files.append(filename)
    return saved_files


# Main Experiment
def main():
    # Step 1: Capture Image
#    original_image = capture_image('original_image.png')
    original_image = 'original_image.png'

    # Step 2: Save in Different Formats
    formats = save_in_formats(original_image, 'test_image')
    
    
    # Step 4: Analyze Images
    data = []
    for img_path in formats:
        quality, size, resolution = analyze_image(img_path)
        data.append({
            'File Name': os.path.basename(img_path),
            'Format': img_path.split('.')[-1],
            'Image Size': resolution,
            'File Size (KB)': round(size, 2),
            'Quality Score': round(quality, 2)
        })
    
    # Step 5: Create a DataFrame and Save Results
    df = pd.DataFrame(data)
    df.to_csv('image_analysis_results.csv', index=False)
    print(df)
    
    # Step 6: Plot Charts
    # Plot File Size vs Format
    plt.figure()
    df[['Format', 'File Size (KB)']].groupby('Format').mean().plot(kind='bar', legend=False)
    plt.ylabel('File Size (KB)')
    plt.title('File Size vs Format')
    plt.savefig('file_size_vs_format.png')
    plt.show()

    # Plot Quality Score vs Format
    plt.figure()
    df[['Format', 'Quality Score']].groupby('Format').mean().plot(kind='bar', legend=False, color='orange')
    plt.ylabel('Quality Score')
    plt.title('Quality Score vs Format')
    plt.savefig('quality_score_vs_format.png')
    plt.show()

# Run the experiment
if __name__ == "__main__":
    main()
