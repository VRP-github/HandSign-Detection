# Hand Sign Detection and Classification

This project uses computer vision to detect and classify hand gestures captured via a webcam. The project consists of two main components: data collection for creating a custom dataset of hand gestures and real-time classification of these gestures using a pre-trained model.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Hand Gesture Classification](#hand-gesture-classification)
---

## Overview

This project provides tools to:
1. Collect images of hand gestures for training a machine learning model.
2. Detect hand gestures in real-time and classify them into predefined categories using a deep learning model.

---

## Features

- **Hand Detection**: Detects hands in the video feed using OpenCV and cvzone's `HandDetector`.
- **Dataset Creation**: Saves cropped hand images for creating a custom dataset.
- **Gesture Classification**: Classifies gestures into categories (e.g., "A", "B", "C", "OK") using a pre-trained Keras model.
- **Real-Time Feedback**: Displays predictions and bounding boxes around detected hands on the live feed.

---

## Dependencies

The project requires the following Python libraries:

- `cv2` (OpenCV)
- `cvzone`
- `numpy`
- `math`

Install these dependencies using pip:
```bash
pip install opencv-python cvzone numpy
```

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the following folders and files are present:
   - `Images/` for storing collected gesture images.
   - `Model/keras_model.h5` for the pre-trained classification model.
   - `Model/labels.txt` for gesture labels.

4. Connect a webcam to your system.

---

## Usage

### Data Collection

1. Run the `dataCollection.py` script to collect gesture images:
   ```bash
   python dataCollection.py
   ```
2. Make gestures in front of the camera. Press `s` to save images of your gesture into the `Images/` folder.
3. Use the saved images to train your custom gesture classification model.

### Hand Gesture Classification

1. Ensure the pre-trained model and labels file are present in the `Model/` folder.
2. Run the `test.py` script to start gesture classification:
   ```bash
   python test.py
   ```
3. Wave your hand in front of the camera to see real-time predictions displayed on the screen.

---
