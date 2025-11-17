👁️ Simple Face Detection using OpenCV

⚙️ Steps
- Import OpenCV
import cv2
- Load Haar Cascade for faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
- Read image and convert to grayscale
img = cv2.imread('face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
- Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
- Draw bounding boxes
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



🔑 Key Notes
- scaleFactor: Controls how much the image is reduced at each scale.
- minNeighbors: Higher values → stricter detection, fewer false positives.
- Works best on frontal faces with good lighting.

👉 Snehal, since you’re already experimenting with object detection pipelines, you can extend this to real-time face detection using your webcam (cv2.VideoCapture(0)). Would you like me to show you that version next so you can test it live in your PG setup?


A beginner-friendly Python project that performs real-time face detection using OpenCV and Haar Cascade classifiers. Ideal for learning computer vision basics and building foundational AI applications.

📦 Features

Real-time face detection via webcam
Uses Haar Cascade classifier
Lightweight and easy to run
Clean bounding box overlay on detected faces
🛠️ Tech Stack
Component	Technology
Language	Python 3.x
Library	OpenCV
Model	Haar Cascade
2. Install Dependencies
pip install opencv-python

3. Run the Script
python face_detect.py

📷 How It Works
Loads Haar Cascade XML model
Captures video from webcam
Detects faces and draws rectangles

🖼️ Simple Object Detection with OpenCV


A lightweight Python project that performs real-time object detection using OpenCV and a pre-trained MobileNet SSD model. Perfect for beginners exploring computer vision and AI fundamentals.

📦 Features
Real-time object detection via webcam
Uses MobileNet SSD with Caffe model
Displays bounding boxes and class labels
Easy to run and modify
🛠️ Tech Stack
Component	Technology
Language	Python 3.x
Library	OpenCV, NumPy
Model	MobileNet SSD
2. Install Dependencies
pip install opencv-python numpy

▶️ Run the Script
python object_detection.py

# Spoof Detection 🕵️‍♂️

A simple project to detect spoofing attacks (e.g., face spoofing using photos, videos, or masks).  
This repository demonstrates basic anti-spoofing techniques using computer vision and machine learning.

## 🚀 Features
- Detects spoof vs. real faces using image/video input
- Supports multiple detection methods (e.g., texture analysis, liveness detection)
- Lightweight and easy to integrate into existing projects
- Modular code structure for experimentation

# Motion Detection 🎥

A simple project to detect motion in video streams using computer vision.  
This repository demonstrates how to capture frames from a webcam or video file and identify changes that indicate movement.



## 🚀 Features
- Detects motion in real-time using webcam or video input
- Highlights moving objects with bounding boxes or contours
- Lightweight and easy to run on any system
- Modular code for experimentation and extension
