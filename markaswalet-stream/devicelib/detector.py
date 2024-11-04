import cv2
import os
from ultralytics import YOLO

YOLO_model = YOLO("model/yolov8n.onnx")

def detect_and_count_birds(img=None, confidence=0.8):

    # Detect objects in the image
    results = YOLO_model.predict(source=[img], conf=confidence, save=False)

    # Get the number of objects detected
    DP = results[0]  # 'results' contains a list of detections

    # Get the class names for 'bird'
    birds = [box for box in DP.boxes if 'bird' in YOLO_model.names[int(box.cls[0])].lower()]

    # Show birds count on the bottom left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        f"Birds: {len(birds)}",
        (10, img.shape[0] - 10),
        font,
        1,
        (255, 60, 125),
        2,
    )

    if len(DP.boxes) == 0:
        return img, 0

    # draw rectangle and label for each bird
    for bird in birds:
        bird_box = bird.xyxy[0].cpu().numpy()  # Bounding box coordinates
        bird_conf = bird.conf.cpu().numpy()[0]  # Confidence score (single value)
        bird_cls = bird.cls.cpu().numpy()[0]  # Class index (single value)
        bird_class_name = YOLO_model.names[int(bird_cls)]  # Class name

        # Draw rectangle and label for each detected bird
        cv2.rectangle(
            img,
            (int(bird_box[0]), int(bird_box[1])),
            (int(bird_box[2]), int(bird_box[3])),
            (0, 255, 0),
            3,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            img,
            f"{bird_class_name} {str(round(float(bird_conf), 3))}%",  # Convert bird_conf to float and round it
            (int(bird_box[0]), int(bird_box[1]) - 10),
            font,
            1,
            (255, 25, 125), # urutan warna BGR
            2,
        )

    
    return img, len(birds)