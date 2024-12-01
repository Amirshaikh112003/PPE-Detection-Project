from ultralytics import YOLO
import cv2

# Step 1: Train the Model
# model = YOLO(r'C:\Users\admin\Desktop\project\runs\detect\train9\weights\best.pt')  # Load Pre-trained YOLOv5 Model

# results = model.train(
#     data='ppe_data.yaml',  # Path to the dataset configuration file
#     epochs=5,             # Number of epochs for training
#     imgsz=320,             # Image size for training
#     batch=4,              # Batch size
#     device='cpu'           # Use CPU for training
# )

# The trained model is automatically saved in 'runs/train/' by YOLO
# You don't need to save it manually, but you can rename it if needed:
# model.save('ppe_detection_model.pt')

# Step 2: Test the Model on Images
model = YOLO(r'C:\Users\admin\Desktop\project\runs\detect\train9\weights\best.pt')  # Load the best model from training

# Predict on test images
# Test the Model on Images
results = model.predict(
    source=r'C:\Users\admin\Desktop\project\dataset\test\images',  # Path to test images
    conf=0.5,               # Confidence threshold for detection
    save=True               # Save the results with bounding boxes
)

# Display each result
for result in results:
    result.show()


# Step 3: Test the Model on a Video
cap = cv2.VideoCapture(r"C:\Users\admin\Desktop\project\sample_video.mp4")  # Replace with '0' for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run PPE Detection on the current frame
    results = model.predict(frame, conf=0.5)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('PPE Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
