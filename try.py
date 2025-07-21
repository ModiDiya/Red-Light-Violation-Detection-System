import torch
import numpy as np
from PIL import Image
import cv2
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from datetime import datetime
import os

# ✅ Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ✅ Check for CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ✅ Load YOLOv8 model
yolo_model = YOLO(r"best (2).pt")
yolo_model.to(device)
class_names = yolo_model.names

# ✅ Load BLIP captioning model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# ✅ Create output directory
output_dir = "vlm_violations"
os.makedirs(output_dir, exist_ok=True)

# ✅ Correct image path (always use raw string or double backslashes or forward slashes)
image_path = r"test\images\212_jpg.rf.8015af9fbfc00763cde6dda5a4829046.jpg"  # <- Update this path if needed

# ✅ Load and preprocess the image
image = Image.open(image_path).convert("RGB").resize((640, 640))

# ✅ Run YOLOv8 detection
results = yolo_model(image_path)[0]

# Init flags
violation_detected = False
detected_label = ""
crop_pil = None
license_plate_text = ""

# ✅ Loop through detected boxes
for box in results.boxes:
    cls = int(box.cls[0])
    label = class_names[cls]
    conf = float(box.conf[0])
    print(f"🔍 Detected {label} with confidence {conf:.2f}")

    if label == "license_plate" and conf > 0.3:  # confidence threshold
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop_pil = image.crop((x1, y1, x2, y2))
        violation_detected = True
        detected_label = label

        # ✅ Convert to OpenCV format
        crop_cv2 = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(crop_cv2, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # ✅ OCR with pytesseract
        license_plate_text = pytesseract.image_to_string(thresh, config='--psm 8')
        print(f"🔠 OCR License Plate Number: {license_plate_text.strip()}")
        break

# ✅ If no license plate found
if not violation_detected:
    print("⚠️ No traffic violation or license plate detected.")
    print("🔎 Generating scene description...")

    # Caption the full image
    prompt = "Describe the contents of this image."
    inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = blip_model.generate(**inputs, max_new_tokens=100)

    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    print(f"📝 BLIP Caption: {caption}")
    exit()

# ✅ Save cropped license plate image
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = os.path.join(output_dir, f"license_plate_{timestamp}.jpg")
crop_pil.save(filename)

# ✅ Generate caption for cropped license plate
prompt = "Describe this traffic violation scene."
inputs = blip_processor(images=crop_pil, text=prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = blip_model.generate(**inputs, max_new_tokens=100)

caption = blip_processor.decode(output[0], skip_special_tokens=True)

# ✅ Final output
print(f"\n✅ Detected Label: {detected_label}")
print(f"📝 BLIP Caption: {caption}")
print(f"📷 Saved Image Path: {filename}")
print(f"🔠 OCR License Plate Number: {license_plate_text.strip()}")               



