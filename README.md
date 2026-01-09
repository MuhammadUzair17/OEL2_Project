# Real-Time Object Classification with AR Overlay (YOLOv8)

This project implements a real-time object detection and classification system using a **YOLOv8 pretrained model** with **Augmented Reality (AR) overlays** on a live webcam feed. Detected objects are shown with bounding boxes, class labels, confidence scores, FPS, and runtime statistics.

---

## What this project does
- Captures live video from webcam
- Preprocesses each frame (resize, color conversion)
- Runs YOLOv8 inference on each frame
- Applies post-processing (confidence filtering, NMS)
- Renders AR overlays (boxes, labels, confidence bars)
- Displays real-time FPS and detection statistics

---

## ML Concepts Integrated
- **Data Preprocessing:** Frame resizing and normalization  
- **Decision Tree & Ensemble Concepts:** YOLO internal prediction logic  
- **ML × AR Integration:** Live ML predictions overlaid on video stream  

---

## Model, Controls, Structure & Notes (All-in-One)

**Model Info**
- Model: YOLOv8 Nano (`yolov8n.pt`)
- Dataset: COCO (80 object classes)
- Confidence Threshold: 0.7
- Input Resolution: 1280 × 720
- Inference Type: Real-time, single-stage detector

**Controls**
- `q` → Quit application  
- `s` → Save current frame  
- `r` → Reset detection statistics  


**Notes**
- Performance depends on hardware and lighting
- Uses pretrained weights (no custom fine-tuning)
- Webcam usage should respect privacy and consent

---

## Installation & Run
**Command Prompt**
- pip install -r requirements.txt
- python real_time_detection.py

