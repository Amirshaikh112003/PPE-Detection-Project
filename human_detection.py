# Import required libraries
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import torch

# Load YOLOv5 model pretrained on COCO dataset (includes human class)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to('cpu')  # This forces the model to use the CPU

# Define No-Human Zones as bounding box coordinates (example zone)
no_human_zones = [(200, 200, 600, 600)]  # (x1, y1, x2, y2) coordinates

# Initialize video capture (0 for webcam or path for video file)
cap = cv2.VideoCapture('C:/Users/admin/Desktop/project/sample_video.mp4')



def is_inside_no_human_zone(x, y, w, h, zones):
    """Check if detected object is within a No-Human Zone."""
    for (zx1, zy1, zx2, zy2) in zones:
        if zx1 < x < zx2 and zy1 < y < zy2:
            return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run the YOLO model on the frame
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Format: [x1, y1, x2, y2, confidence, class]

    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        
        # Check if detected object is a person (class 0 for person in COCO dataset)
        if cls == 0:  
            color = (0, 255, 0)
            label = "Person"
            
            # Check if the person is in a No-Human Zone
            if is_inside_no_human_zone(x1, y1, x2 - x1, y2 - y1, no_human_zones):
                color = (0, 0, 255)  # Change color to red for alert
                label += " - In Restricted Zone!"
                cv2.putText(frame, "Alert: Unauthorized Access", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw No-Human Zone boundaries
    for (x1, y1, x2, y2) in no_human_zones:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "No-Human Zone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("PPE Detection", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
