import cv2
import torch
import numpy as np
from PIL import Image
import pytesseract
from ultralytics import YOLO
from datetime import datetime
import os

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load YOLOv8 model
model = YOLO("best (2).pt").to(device)
class_names = model.names

# Input video path
video_path = r"3.mp4"

# Output directory
output_dir = "violations_output"
os.makedirs(output_dir, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
frame_interval = 30
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    print(f"\n📸 Processing frame {frame_count}")

    # Resize to match YOLO input (if needed)
    resized_frame = cv2.resize(frame, (640, 640))
    pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

    # Run detection
    results = model(pil_image)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        label = class_names[cls]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label in ["license_plate", "NoHelmet", "red_light"] and conf > 0.4:
            print(f"🔍 Detected {label} (conf: {conf:.2f})")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

            # Draw rectangle and label
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(resized_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Crop and save image
            cropped_pil = pil_image.crop((x1, y1, x2, y2))
            save_path = os.path.join(output_dir, f"{label}_{timestamp}.jpg")
            cropped_pil.save(save_path)

            if label == "license_plate":
                # OCR
                crop_cv = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                text = pytesseract.image_to_string(thresh, config='--psm 8').strip()
                print(f"🔠 OCR Text: {text}")

                # Draw OCR result
                if text:
                    cv2.putText(resized_frame, f"OCR: {text}", (x1, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Violation Detection", resized_frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Video processing completed.")
