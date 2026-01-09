# Real-Time Object Classifier - Webcam Detection 
# YOLOv8 Pretrained Model Real-Time Detection with AR Overlay


# STEP 1: Import Libraries
import cv2
import time
from ultralytics import YOLO
import numpy as np
import os


# STEP 2: Configuration
MODEL_NAME = 'yolov8n.pt'  

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.7

# Camera settings
CAMERA_INDEX = 0  # 0 for default camera
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Colors for bounding boxes (BGR format)
COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (255, 165, 0),    # Orange
    (0, 128, 128),    # Teal
    (128, 128, 0)     # Olive
]


# STEP 3: Load YOLOv8 Pretrained Model
print("\n")
print("Real-Time Object Detection with AR Overlay")
print("Using YOLOv8 Pretrained Model (80 COCO Classes)")
print("\n")
print("\nLoading YOLOv8 pretrained model.")
print("This will download the model if not already present.")

try:
    model = YOLO(MODEL_NAME)
    print(" Model loaded successfully!")
    print(f"Model has {len(model.names)} classes (COCO dataset)")
except Exception as e:
    print(f" Error loading model: {e}")
    print("\nMake sure you have installed all dependencies:")
    print("  pip install ultralytics")
    print("  pip install opencv-python==4.8.1.78")
    print("  pip install torch==2.0.1")
    print("  pip install torchvision==0.15.2")
    exit()

# STEP 4: Initialize Webcam
print("\nInitializing webcam")
cap = cv2.VideoCapture(CAMERA_INDEX)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Check if camera opened successfully
if not cap.isOpened():
    print("✗ Error: Could not open webcam")
    print("\nTroubleshooting:")
    print("  1. Check if camera is connected")
    print("  2. Close other apps using the camera")
    print("  3. Try changing CAMERA_INDEX to 1 or 2")
    exit()

print(" Webcam initialized successfully!")

# STEP 5: Performance Tracking Variables
# FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

# Statistics
total_detections = 0
detection_counts = {}

# STEP 6: AR Overlay Function
def draw_ar_overlay(frame, boxes, confidences, class_ids, class_names):
    """
    Draw AR overlay with bounding boxes and labels on detected objects
    
    Args:
        frame: Input frame
        boxes: List of bounding boxes [x1, y1, x2, y2]
        confidences: List of confidence scores
        class_ids: List of class IDs
        class_names: Dictionary of class names
    
    Returns:
        frame: Frame with AR overlay
    """
    global total_detections, detection_counts
    
    for i, box in enumerate(boxes):
        # Extract box coordinates
        x1, y1, x2, y2 = map(int, box)
        
        # Get class info
        class_id = int(class_ids[i])
        confidence = confidences[i]
        class_name = class_names[class_id]
        
        # Update statistics
        total_detections += 1
        if class_name in detection_counts:
            detection_counts[class_name] += 1
        else:
            detection_counts[class_name] = 1
        
        # Select color based on class
        color = COLORS[class_id % len(COLORS)]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = f"{class_name} {confidence:.2f}"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw text label
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Draw confidence bar
        bar_width = int((x2 - x1) * confidence)
        cv2.rectangle(
            frame,
            (x1, y2 + 5),
            (x1 + bar_width, y2 + 15),
            color,
            -1
        )
    
    return frame

# STEP 7: Display Performance Metrics
def draw_performance_metrics(frame, fps, detection_count):
    """
    Display performance metrics on frame
    
    Args:
        frame: Input frame
        fps: Current FPS
        detection_count: Number of detections in current frame
    
    Returns:
        frame: Frame with metrics overlay
    """
    # Draw semi-transparent background for metrics
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    
    # Draw detection count
    cv2.putText(
        frame,
        f"Detections: {detection_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    
    # Draw total detections
    cv2.putText(
        frame,
        f"Total: {total_detections}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    
    # Draw instructions
    cv2.putText(
        frame,
        "Press 'q' to quit | 's' to save frame | 'r' to reset stats",
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )
    
    return frame

# STEP 8: Main Detection Loop
print("\n")
print("Real-Time Object Detection Started")
print("\n")
print("\nControls:")
print("  • Press 'q' to quit")
print("  • Press 's' to save current frame")
print("  • Press 'r' to reset statistics")
print("\nDetecting 80 COCO classes:")
print("  person, bicycle, car, motorcycle, airplane, bus, train,")
print("  truck, boat, bottle, cup, fork, knife, spoon, bowl, chair,")
print("  laptop, mouse, keyboard, cell phone, book, clock, and more...")
print("\n")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Increment frame count
    frame_count += 1
    
    # Calculate FPS every 30 frames
    if frame_count % 30 == 0:
        end_time = time.time()
        fps = 30 / (end_time - start_time)
        start_time = time.time()
    
    # STEP 9: Run Object Detection
    # Run YOLOv8 inference
    results = model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )
    
    # Extract detection results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs
    class_names = results[0].names  # Class names dictionary
    
    # STEP 10: Draw AR Overlay
    if len(boxes) > 0:
        frame = draw_ar_overlay(frame, boxes, confidences, class_ids, class_names)
    
    # STEP 11: Display Performance Metrics
    frame = draw_performance_metrics(frame, fps, len(boxes))
    
    # STEP 12: Display Frame
    cv2.imshow('Real-Time Object Detection with AR Overlay', frame)
    
    # STEP 13: Handle Keyboard Input
    key = cv2.waitKey(1) & 0xFF
    
    # Quit on 'q' key
    if key == ord('q'):
        print("\n")
        print("Shutting down...")
        print("\n")
        break
    
    # Save frame on 's' key
    elif key == ord('s'):
        # Create screenshots folder if it doesn't exist
        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")
    
    # Reset statistics on 'r' key
    elif key == ord('r'):
        total_detections = 0
        detection_counts = {}
        frame_count = 0
        start_time = time.time()
        print(" Statistics reset")

# STEP 14: Cleanup and Display Final Statistics
# Release resources
cap.release()
cv2.destroyAllWindows()

# Calculate final metrics
total_time = time.time() - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0

# Display final statistics
print("\n")
print("Final Statistics")
print("\n")
print(f"Total Frames Processed: {frame_count}")
print(f"Total Detections: {total_detections}")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Total Runtime: {total_time:.2f} seconds")

if len(detection_counts) > 0:
    print("\nDetection Breakdown (Top 10):")
    sorted_detections = sorted(detection_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (class_name, count) in enumerate(sorted_detections[:10], 1):
        print(f"  {i}. {class_name}: {count}")
else:
    print("\nNo objects detected")

print("\n")
print("Real-time detection stopped successfully!")
