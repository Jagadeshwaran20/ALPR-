import os
import cv2
import time
from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
from paddleocr import PaddleOCR
import uuid

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
model = YOLO("D:/anpr_model/weights/best.pt")  # Your trained YOLOv8 model path
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Process frame function
def process_frame(frame):
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    for box in boxes:
        x1, y1, x2, y2 = box
        plate_img = frame[y1:y2, x1:x2]
        plate_text = "No text detected"
        try:
            result = ocr.ocr(plate_img, cls=True)
            if result and isinstance(result[0], list) and len(result[0]) > 0:
                plate_text = " ".join([line[1][0] for line in result[0] if line])
        except Exception as e:
            print("OCR error:", e)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return frame

# Homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'media' not in request.files:
            return "No file uploaded", 400

        file = request.files['media']
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Process image
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            frame = cv2.imread(filepath)
            if frame is None:
                return "Failed to read uploaded image", 400
            processed = process_frame(frame)
            output_path = os.path.join(UPLOAD_FOLDER, "processed_" + filename)
            cv2.imwrite(output_path, processed)
            return render_template("index.html", input_image=filepath, output_image=output_path)

        return "Only image files are supported right now (jpg, png).", 415

    return render_template("index.html")
if __name__ == '__main__':
    app.run(debug=True)
