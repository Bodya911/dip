# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 13:47:29 2025

@author: bzh_p
"""

import cv2
import os
from brisque import BRISQUE
import matplotlib.pyplot as plt
import pandas as pd

# Function to record video and save in multiple formats
def record_video(output_prefix, duration=5, fps=30, resolution=(640, 480)):
    cap = cv2.VideoCapture(0)  # Start the webcam
    fourcc_formats = {
        "i420 (avi)": cv2.VideoWriter_fourcc(*"I420"),
        "mp4": cv2.VideoWriter_fourcc(*"mp4v"),
        "DivX (avi)": cv2.VideoWriter_fourcc(*"DIVX"),
        "flv": cv2.VideoWriter_fourcc(*"FLV1"),
        "x264 (avi)": cv2.VideoWriter_fourcc(*"XVID"),
    }
    video_files = []
    frame_count = duration * fps

    for format_name, fourcc in fourcc_formats.items():
        filename = f"{output_prefix}_{format_name.replace(' ', '_').replace('(', '').replace(')', '')}.avi"
        video_files.append(filename)
        out = cv2.VideoWriter(filename, fourcc, fps, resolution)
        print(f"Recording {format_name} video...")

        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()

    cap.release()
    cv2.destroyAllWindows()
    return video_files, resolution, fps, duration

# Function to extract a frame from a video and save it as an image
def extract_frames(video_files, duration, fps, resolution, output_format="png"):
    extracted_frames = []
    obj = BRISQUE(url=False)

    for video in video_files:
        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()  # Read the first frame
        if ret:
            output_image = video.replace(".avi", f".{output_format}")
            cv2.imwrite(output_image, frame)
            quality_score = obj.score(frame)
            file_size = os.path.getsize(video) / 1024  # File size in KB

            # Extract the format name from the filename (e.g., "i420" from "test_video_i420_avi.avi")
            format_name = video.split('_')[2]  # Assuming format is always at index 2 in the filename

            extracted_frames.append({
                "Video File": os.path.basename(video),
                "Format": format_name,
                "Duration (s)": duration,
                "FPS": fps,
                "Resolution": f"{resolution[0]}x{resolution[1]}",
                "File Size (KB)": round(file_size, 2),
                "Extracted Frame Quality": round(quality_score, 2),
                "Extracted Frame": output_image
            })
        cap.release()
    
    return extracted_frames


# Function to plot results
def plot_results(df, x_column, y_column, title, ylabel, output_file, color="blue"):
    plt.figure()
    df.plot(x=x_column, y=y_column, kind="bar", legend=False, color=color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(x_column)
    plt.savefig(output_file)
    plt.show()

# Main Experiment Workflow
def main():
    # Step 1: Record Videos
    output_prefix = "test_video"
    duration = 5  # in seconds
    fps = 30
    resolution = (640, 480)
    video_files, resolution, fps, duration = record_video(output_prefix, duration, fps, resolution)
    print("Videos recorded successfully!")

    # Step 2: Extract Frames and Analyze
    frame_data = extract_frames(video_files, duration, fps, resolution)
    df_frames = pd.DataFrame(frame_data)
    df_frames.to_csv("video_analysis_results.csv", index=False)
    print("Frame extraction and analysis complete!")
    print(df_frames)

    # Step 3: Plot Results
    plot_results(
        df=df_frames,
        x_column="Format",
        y_column="File Size (KB)",
        title="File Size vs Video Format",
        ylabel="File Size (KB)",
        output_file="file_size_vs_video_format.png"
    )

    plot_results(
        df=df_frames,
        x_column="Format",
        y_column="Extracted Frame Quality",
        title="Extracted Frame Quality vs Video Format",
        ylabel="Quality Score",
        output_file="quality_vs_video_format.png",
        color="orange"
    )

# Run the Experiment
if __name__ == "__main__":
    main()
