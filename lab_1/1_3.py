# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:42:59 2025

@author: bzh_p
"""

import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from brisque import BRISQUE

# Function to convert color images to grayscale (3 channels)
def convert_to_grayscale_3_channels(color_image_path, output_prefix):
    color_image = cv2.imread(color_image_path)
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)  # Convert to single-channel grayscale
    grayscale_3_channel = cv2.merge([grayscale_image, grayscale_image, grayscale_image])  # Expand to 3 channels
    
    grayscale_files = []
    formats = ['bmp', 'jpg', 'png']
    for fmt in formats:
        output_path = f"{output_prefix}_gray.{fmt}"
        cv2.imwrite(output_path, grayscale_3_channel)
        grayscale_files.append(output_path)
    
    return grayscale_files

# Function to analyze images (file size and quality)
def analyze_images(image_files):
    obj = BRISQUE(url=False)
    data = []
    for image_path in image_files:
        image = cv2.imread(image_path)
        quality_score = obj.score(image)
        file_size = os.path.getsize(image_path) / 1024  # File size in KB
        data.append({
            "Image File": os.path.basename(image_path),
            "Format": image_path.split('.')[-1],
            "File Size (KB)": round(file_size, 2),
            "Quality Score": round(quality_score, 2)
        })
    return data

# Function to plot comparison charts
def plot_comparison_chart(df, x_column, y_column, title, ylabel, output_file, color="blue"):
    plt.figure()
    df.plot(x=x_column, y=y_column, kind="bar", color=color, legend=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(x_column)
    plt.savefig(output_file)
    plt.show()

# Main Workflow for Grayscale Experiment
def main():
    # Step 1: Convert color images to grayscale and save
    color_image_files = ["test_image.bmp", "test_image.jpg", "test_image.png"]  # Replace with actual color image file paths
    grayscale_files = []
    for color_image in color_image_files:
        output_prefix = color_image.split('.')[0]
        grayscale_files += convert_to_grayscale_3_channels(color_image, output_prefix)
    
    print("Grayscale images created successfully!")
    
    # Step 2: Analyze color and grayscale images
    color_data = analyze_images(color_image_files)
    grayscale_data = analyze_images(grayscale_files)

    # Combine results into a single DataFrame
    df_color = pd.DataFrame(color_data)
    df_grayscale = pd.DataFrame(grayscale_data)
    df_color["Image Type"] = "Color"
    df_grayscale["Image Type"] = "Grayscale"
    df_combined = pd.concat([df_color, df_grayscale])
    df_combined.to_csv("grayscale_image_analysis.csv", index=False)
    print(df_combined)

    # Step 3: Plot comparison charts
# File size comparison (fix duplicate issue by aggregating)
    file_size_df = df_combined.groupby(["Format", "Image Type"])["File Size (KB)"].mean().unstack()
    file_size_df.plot(kind="bar", figsize=(8, 5))
    plt.title("File Size: Color vs Grayscale")
    plt.ylabel("File Size (KB)")
    plt.xlabel("Format")
    plt.savefig("file_size_comparison.png")
    plt.show()
    
    # Quality score comparison (fix duplicate issue by aggregating)
    quality_score_df = df_combined.groupby(["Format", "Image Type"])["Quality Score"].mean().unstack()
    quality_score_df.plot(kind="bar", figsize=(8, 5), color=["blue", "orange"])
    plt.title("Quality Score: Color vs Grayscale")
    plt.ylabel("Quality Score")
    plt.xlabel("Format")
    plt.savefig("quality_score_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
