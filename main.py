import cv2
import os
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Load YOLOv8 model (change path if needed)
model = YOLO("D:/anpr_model/weights/best.pt")

# Load PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # For English license plates

# Function to process individual frames (for both image and video)
def process_frame(frame, do_ocr=True):
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    for box in boxes:
        x1, y1, x2, y2 = box
        plate_img = frame[y1:y2, x1:x2]

        plate_text = "No text detected"
        if do_ocr:
            result = ocr.ocr(plate_img, cls=True)
            if result and isinstance(result[0], list) and len(result[0]) > 0:
                try:
                    plate_text = " ".join([line[1][0] for line in result[0] if line])
                except Exception as e:
                    print("OCR parsing error:", e)
                    plate_text = "OCR error"

        # Draw detection box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return frame

# Function for processing a single image
def run_on_image(image_path):
    assert os.path.exists(image_path), "Image file not found!"
    frame = cv2.imread(image_path)
    processed = process_frame(frame, do_ocr=True)
    cv2.imshow("ALPR - Image", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function for processing a video
def run_on_video(video_path):
    assert os.path.exists(video_path), "Video file not found!"
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame, do_ocr=True)
        cv2.imshow("ALPR - Video", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# MAIN ENTRY POINT
if __name__ == "__main__":
    # Provide the full path to image or video
    media_path = "D:/alpr code/output/demo.jpg"  # Replace with your test image or video

    if media_path.lower().endswith((".jpg", ".png", ".jpeg")):
        run_on_image(media_path)
    elif media_path.lower().endswith((".mp4", ".avi", ".mov")):
        run_on_video(media_path)
    else:
        print("Unsupported file format.")
