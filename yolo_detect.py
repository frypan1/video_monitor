import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model (Use a more accurate model)
model = YOLO("models/yolov8m.pt")  # Use 'yolov8l.pt' for better accuracy

# YOLO label for a person
PERSON_LABEL = "person"

# Confidence threshold for detection
CONFIDENCE_THRESHOLD = 0.4

# Video input/output setup
video_path = "videos/input1.mp4"
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(1)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = "output/output.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

last_classification = "Unknown"

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run YOLO detection
    annotated_frame = results[0].plot()  # Draw bounding boxes

    # Track detected objects in the current frame
    detected_objects = set()
    person_count = 0

    for box in results[0].boxes:
        if float(box.conf[0]) > CONFIDENCE_THRESHOLD:
            obj_name = model.names[int(box.cls[0])]
            detected_objects.add(obj_name)
            if obj_name == PERSON_LABEL:
                person_count += 1

    # Determine classification
    if person_count == 1 and len(detected_objects) == 1:  # Only one person, no other objects
        last_classification = "Professional"
    else:
        last_classification = "Unprofessional"  # More than 1 person or any other object present

    # Display classification on the video frame
    color = (0, 255, 0) if last_classification == "Professional" else (0, 0, 255)
    cv2.putText(annotated_frame, f"Environment: {last_classification}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Show video output
    cv2.imshow("Monitoring Interview Environment", annotated_frame)
    out.write(annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
