🚦 Traffic Violation Detection System
An AI-powered system that detects traffic violations such as red light jumping, helmet-less riding, and wrong-side driving using YOLOv8. It logs violations with timestamps and provides a simple web interface using Flask for real-time video analysis.
An AI-powered computer vision system that automatically detects traffic violations such as **red light jumping**, **helmetless riding**, **wrong-side driving**, and optionally performs **license plate recognition** using YOLOv8.
![Traffic Violation Banner](https://via.placeholder.com/1000x300.png?text=Traffic+Violation+Detection+System

📌 Project Overview

This project aims to enhance road safety by detecting and recording common traffic violations using real-time video or image input. It uses a trained YOLOv8 object detection model and Flask web framework to provide a simple UI for users to upload videos or images and view results with bounding boxes, captions, and logs.


🎯 Key Features

- 🚦 Red Light Violation Detection
- 🛑 Stop Line Detection
- 🪖 Helmet & No Helmet Detection
- 🔄 Wrong Side Driving Detection
- 🔍 License Plate Recognition (OCR)
- 📁 Image and Video Upload Support
- 📊 Violation Logs with Time & Snapshot

🧠 Tech Stack

| Layer        | Technology                          |
|--------------|-------------------------------------|
| Frontend     | HTML, CSS, Jinja2 (Flask Templates) |
| Backend      | Python, Flask                       |
| CV/AI Model  | YOLOv8 (Ultralytics)                |
| OCR Engine   | Tesseract or EasyOCR                |
| Deployment   | Localhost (Flask) or Web Server     |

🔮 Future Scope

Integrate with live traffic camera feeds for real-time monitoring.
Automate e-challan generation using license plate detection.
Deploy on cloud or edge devices (e.g., Jetson Nano) for scalable use.
Add analytics dashboard for violation trends and hotspots.
Extend system with features like speed detection and mobile alerts.

🗂️ Project Structure
traffic_violation_app/
│
├── yolov8_model/ # Trained YOLOv8 model (best.pt)
├── static/
│ ├── uploads/ # Uploaded videos/images
│ ├── results/ # YOLOv8 detection outputs
│ └── vlm/ # Cropped violation images
├── templates/
│ ├── index.html # Upload form
│ └── result.html # Result display
├── app.py # Flask app
└── README.md

 🚀 How to Run

 1. Clone the Repository

```bash
git clone https://github.com/your-username/traffic-violation-detection.git
cd traffic-violation-detection

2. Create Virtual Environment (Optional but Recommended)
bash
python -m venv venv
.\venv\Scripts\activate   # On Windows

3. Install Requirements
bash
pip install flask ultralytics opencv-python pillow numpy pytesseract transformers torch

4. Run the App
bash
python app.py
Open your browser and go to: http://127.0.0.1:5000/

🧪 Model Classes
python
Copy
Edit
['car', 'bus', 'bike', 'helmet', 'no_helmet', 'red_light', 'stop_line', 'wrong_side', 'license_plate']


🙋‍♀️ Developed By
Diya Modi
B.Tech CSE, Navrachana University
🌐 LinkedIn | ✉️ Email


