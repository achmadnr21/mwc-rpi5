import numpy as np
import cv2
from picamera2 import Picamera2

from datetime import datetime

IMSIZE = (640, 480)
FPS = 15  # Adjust to your desired frame rate

# Initialize the PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": IMSIZE}))
picam2.start()

cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)


detector = cv2.CascadeClassifier('haarcascadeku/haarcascade_frontalface_default.xml')
def get_faces(image = None):
    # global detector
    # if detector is None:
    #    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Converting to grey")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray)

    # bounding box
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return faces, len(faces)


try:
    while True:
        frame = picam2.capture_array()
        if frame is None:
            break
        
        # Convert to BGR format and flip the frame
        frame_bgr = np.asarray(frame[:, :, 0:3], dtype=np.uint8)
        frame_bgr = cv2.flip(frame_bgr, -1)
        new_frame_bgr, count = get_faces(frame_bgr)

        current_time = datetime.now().strftime("%H:%M:%S")
        # Write it
        frame_rgb = cv2.cvtColor(new_frame_bgr, cv2.COLOR_BGR2RGB)

        # Display the frame in the OpenCV window
        cv2.imshow("Camera Feed", frame_rgb)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop the camera and close the window
    picam2.stop()
    cv2.destroyAllWindows()
