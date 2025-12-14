# 🏋️ Deep Learning Fitness Analysis Assistant

## Project Overview

This project is a real-time computer vision system designed to analyze and correct exercise form using **Instance Segmentation (YOLOv8)** and **Geometric Analysis (OpenCV)**.

The application captures video from a webcam, isolates the user's body, calculates the center of mass (Centroid) and the body's rotation (Orientation), and provides immediate coaching feedback and post-session movement analysis.

### Key Features

* **Real-Time Segmentation:** Uses the YOLOv8-Seg model to create a precise, irregular mask of the user's body, enabling accurate analysis.
* **Geometric Analysis:** Calculates the object's Centroid ($C_x, C_y$) and Orientation ($\theta$) using **Image Moments** to quantify position and rotational stability.
* **Real-Time Form Correction:** Displays a "FORM OK" or **"TILT WARNING!"** message based on whether the body's angle falls outside acceptable limits (e.g., $70^\circ$ to $110^\circ$).
* **Path Tracking & Visualization:** Logs the Centroid position during the exercise and generates a `matplotlib` graph of the lift path upon completion.

## 🚀 Getting Started

### Prerequisites

You need Python 3.8+ and the following libraries. It is highly recommended to use a virtual environment (`venv`) for installation.

``bash
# 1. Install core dependencies
pip install numpy opencv-python matplotlib

# 2. Install the Deep Learning Framework (YOLOv8)

pip install ultralytics

ExecutionSave the entire Python script (your final code) as fitness_analysis.py.Run the script from your terminal:Bashpython fitness_analysis.py

The webcam feed will open. 

Perform your exercise (e.g., a squat, bicep curl, or simply rotating an object).

Press the 'q' key to stop the analysis.A separate window will pop up showing the Centroid Path Tracking Analysis graph.

## 🧠 The Technical Breakdown

The system relies on a three-stage pipeline to analyze and correct exercise form.

| Stage | Key Tool / Math | Purpose | Output |
| :--- | :--- | :--- | :--- |
| **1. Isolation** | **YOLOv8 Instance Segmentation** | Finds and isolates the user's body in the video frame, ignoring background. | Precise, irregular **Mask** (white blob) |
| **2. Analysis** | **Image Moments** ($\mathbf{M_{ij}}$ and $\mathbf{\mu_{ij}}$) | Converts the shape of the mask into quantitative, objective metrics. | Centroid ($C_x, C_y$) and Orientation ($\theta$) |
| **3. Coaching** | **Conditional Logic** (`if/else` checks) | Compares the measured Orientation ($\theta$) against the safe range ($70^\circ$ to $110^\circ$). | Real-Time Feedback ("FORM OK" / "TILT WARNING!") |
