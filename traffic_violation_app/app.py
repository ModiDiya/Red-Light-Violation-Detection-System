# from flask import Flask, render_template, request, redirect, url_for
# from ultralytics import YOLO
# import os
# from datetime import datetime



# app = Flask(__name__)
# UPLOAD_FOLDER = 'static/uploads'
# RESULT_FOLDER = 'static/results'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)

# model = YOLO('yolov8_model/best.pt')

# def is_video(filename):
#     video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
#     return filename.lower().endswith(video_extensions)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/detect', methods=['POST'])
# def detect():
#     file = request.files.get('media')
#     if file:
#         timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
#         filename = f"{timestamp}_{file.filename}"
#         filepath = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(filepath)

#         # Run YOLO and get results
#         results = model(filepath, save=True, project=RESULT_FOLDER, name='runs', exist_ok=False)

#         # results.save() saves the file to a subdirectory, like static/results/runs/exp/
#         # We fetch the exact output file from there
#         result_dir = results[0].save_dir  # e.g., static/results/runs/exp/
#         result_name = os.path.basename(results[0].path)  # e.g., same filename
#         result_path = os.path.join(result_dir, result_name)

#         # Relativize for HTML rendering
#         # input_img = os.path.relpath(filepath, 'static')
#         input_img = os.path.relpath(filepath, 'static').replace('\\', '/')
#         # result_img = os.path.relpath(result_path, 'static')
#         result_img = os.path.relpath(result_path, 'static').replace('\\', '/')

#         return render_template('result.html', input_img=input_img, result_img=result_img)
    

#     return redirect(url_for('home'))

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import os
from datetime import datetime
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO('yolov8_model/best.pt')

def is_video(filename):
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    return filename.lower().endswith(video_extensions)

def convert_avi_to_mp4(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get('media')
    if file:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Run detection
        results = model(filepath, save=True, project=RESULT_FOLDER, name='runs', exist_ok=False)

        result_dir = results[0].save_dir  # e.g., static/results/runs/exp/
        result_name = os.path.basename(results[0].path)  # e.g., filename.avi
        result_path = os.path.join(result_dir, result_name)

        # Convert .avi to .mp4 if needed
        if result_path.endswith('.avi'):
            mp4_path = result_path.replace('.avi', '.mp4')
            convert_avi_to_mp4(result_path, mp4_path)
            os.remove(result_path)  # Optional: delete the .avi file
            result_path = mp4_path

        # Clean path for rendering
        input_img = os.path.relpath(filepath, 'static').replace('\\', '/')
        result_img = os.path.relpath(result_path, 'static').replace('\\', '/')

        return render_template('result.html', input_img=input_img, result_img=result_img)

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, render_template, request, redirect, url_for
# from ultralytics import YOLO
# from datetime import datetime
# import os
# import cv2
# from PIL import Image
# import numpy as np
# import pytesseract
# import torch
# from transformers import BlipProcessor, BlipForConditionalGeneration
# pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


# app = Flask(__name__)
# UPLOAD_FOLDER = 'static/uploads'
# RESULT_FOLDER = 'static/results'
# VLM_FOLDER = 'static/vlm'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)
# os.makedirs(VLM_FOLDER, exist_ok=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)

# # Load models
# model = YOLO('yolov8_model/best.pt')
# model.to(device)
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# def is_video(filename):
#     return filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))

# def convert_avi_to_mp4(input_path, output_path):
#     cap = cv2.VideoCapture(input_path)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out.write(frame)
#     cap.release()
#     out.release()

# def process_violation(image_path):
#     image = Image.open(image_path).convert("RGB").resize((640, 640))
#     cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#     results = model(image_path)[0]
#     class_names = model.names

#     violation_detected = False
#     crop_pil = None
#     detected_label = ""
#     license_plate = ""

#     for box in results.boxes:
#         cls = int(box.cls[0])
#         label = class_names[cls]
#         x1, y1, x2, y2 = map(int, box.xyxy[0])

#         if label in ["NoHelmet", "red_light"] and not violation_detected:
#             crop_pil = image.crop((x1, y1, x2, y2))
#             detected_label = label
#             violation_detected = True

#         elif label == "license_plate":
#             plate_crop = cv_image[y1:y2, x1:x2]
#             gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
#             license_plate = pytesseract.image_to_string(gray, config='--psm 7').strip()

#     caption = ""
#     cropped_path = ""
#     if violation_detected and crop_pil:
#         # Save cropped image
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         cropped_path = os.path.join(VLM_FOLDER, f"violation_{timestamp}.jpg")
#         crop_pil.save(cropped_path)

#         # Generate BLIP caption
#         inputs = blip_processor(images=crop_pil, text="Describe this traffic violation scene.", return_tensors="pt").to(device)
#         with torch.no_grad():
#             output = blip_model.generate(**inputs, max_new_tokens=100)
#         caption = blip_processor.decode(output[0], skip_special_tokens=True)

#     return {
#         "detected_label": detected_label,
#         "license_plate": license_plate,
#         "caption": caption,
#         "cropped_img": os.path.relpath(cropped_path, 'static').replace("\\", "/") if cropped_path else ""
#     }

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/detect', methods=['POST'])
# def detect():
#     file = request.files.get('media')
#     if file:
#         timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
#         filename = f"{timestamp}_{file.filename}"
#         filepath = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(filepath)

#         if is_video(filename):
#             results = model(filepath, save=True, project=RESULT_FOLDER, name='runs', exist_ok=False)
#             result_dir = results[0].save_dir
#             result_name = os.path.basename(results[0].path)
#             result_path = os.path.join(result_dir, result_name)
#             if result_path.endswith('.avi'):
#                 mp4_path = result_path.replace('.avi', '.mp4')
#                 convert_avi_to_mp4(result_path, mp4_path)
#                 os.remove(result_path)
#                 result_path = mp4_path
#             result_img = os.path.relpath(result_path, 'static').replace('\\', '/')
#             input_img = os.path.relpath(filepath, 'static').replace('\\', '/')
#             return render_template('result.html', input_img=input_img, result_img=result_img)

#         else:
#             # Process image using VLM pipeline
#             vlm_result = process_violation(filepath)
#             input_img = os.path.relpath(filepath, 'static').replace('\\', '/')
#             return render_template(
#                 'result.html',
#                 input_img=input_img,
#                 result_img=vlm_result.get("cropped_img", ""),
#                 caption=vlm_result.get("caption", ""),
#                 license_plate=vlm_result.get("license_plate", ""),
#                 violation_type=vlm_result.get("detected_label", "")
#             )

#     return redirect(url_for('home'))

# if __name__ == '__main__':
#     app.run(debug=True)
