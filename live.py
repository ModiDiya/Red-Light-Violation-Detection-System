from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import csv

# Load YOLOv8 model
model = YOLO(r"best (2).pt")

# Class names (same as your model)
class_names = [
    'Auto', 'Bus', 'Cycle', 'Helmet', 'Load Auto', 'LoadRickshaw', 'NoHelmet', 'Pedestrian',
    'Van', 'ambulance', 'auto-rikshaw', 'bike', 'bus', 'car', 'green_light', 'helmet',
    'license_plate', 'motobike', 'motorcyclist', 'police vehicle', 'red_light', 'stop_line',
    'tempo', 'truck', 'yellow_light'
]

# List of vehicle-related classes
vehicle_classes = [
    'car', 'motorcyclist', 'bus', 'truck', 'motobike', 'bike', 'auto-rikshaw',
    'Auto', 'Load Auto', 'LoadRickshaw', 'tempo', 'Van'
]

# Output directories
output_dir = "violations"
os.makedirs(output_dir, exist_ok=True)

# CSV log file
log_file = os.path.join(output_dir, "violation_log.csv")
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Violation_Image"])

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)
frame_width, frame_height = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Save video
out = cv2.VideoWriter("live_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (frame_width, frame_height))

# Red light violation counter
red_light_violations = 0

print("[INFO] Starting live feed. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame.")
        break

    # Inference
    results = model(frame, verbose=False)[0]

    red_light_detected = False
    vehicle_detected = False

    # Process detections
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            label = class_names[class_id]

            # Draw boxes and labels
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Detect red light and vehicle presence
            if label == "red_light":
                red_light_detected = True
            if label in vehicle_classes:
                vehicle_detected = True

    # Violation condition: vehicle crossed red light
    if red_light_detected and vehicle_detected:
        red_light_violations += 1
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(output_dir, f"violation_{timestamp}.jpg")
        cv2.imwrite(filename, frame)

        # Log violation
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, filename])

    # Display violation count
    cv2.putText(frame, f"Red Light Violations: {red_light_violations}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    # Show output
    out.write(frame)
    cv2.imshow("Live Traffic Monitoring", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("[INFO] Program ended.")
