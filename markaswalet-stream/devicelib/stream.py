# in this file, we define the Stream procedure to perform the streaming process
import numpy as np
import cv2
from picamera2 import Picamera2
from datetime import datetime
import time
import subprocess
import os
# import devicelib.detector as detector

# import raspberry pi pins to pull up digital pin for relay

import gpiod

IRLED= 16
# start time default is 17:00 and turn it off next day at 5:00
def relay_on_time_between(LED_LINE = None):
    start_time = 17
    end_time = 5
    current_time = datetime.now().hour
    if current_time >= start_time or current_time <= end_time:
        LED_LINE.set_value(0)
    else:
        LED_LINE.set_value(1)

TIMER = time.time()
def write_image(frame=None):
    global TIMER  # Menggunakan variabel global TIMER
    current_time = time.time()  # Mendapatkan waktu saat ini

    if current_time - TIMER >= 30:
        ds_dir = os.path.expanduser('~/dataset')
        print(f'saving : {ds_dir}')
        if not os.path.exists(ds_dir):
            os.makedirs(ds_dir)

        # Mendapatkan timestamp
        timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')

        # Membuat nama file
        file_name = f'image_{timestamp}.jpg'

        # Menyimpan gambar ke direktori
        cv2.imwrite(f'{ds_dir}/{file_name}', frame)

        # Memperbarui TIMER setelah menyimpan gambar
        TIMER = current_time
# get faces
detector = cv2.CascadeClassifier('haarcascadeku/haarcascade_frontalface_default.xml')
def get_faces(image = None):
    # global detector
    # if detector is None:
    #    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray)

    # bounding box
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image, len(faces)
# Stream procedure

IMSIZE = (640, 480)
FPS = 15  # Adjust to your desired frame rate

# Initialize the PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": IMSIZE}))


def stream_process(stream_ip = '103.193.179.252' ,stream_key='mwcdef'):
    # jika stream_key == 'mwcdef' maka akan melakukan EXIT dari program
    if stream_key == 'mwcdef' or stream_key is None:
        print('Exiting from the program')
        exit(0)
    
    # add         '-loglevel', 'error', to shut up the log
    # New preset
    command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-y',  # overwrite output files
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',  # Changed to rgb24 to match XRGB8888 format
        '-s', f'{IMSIZE[0]}x{IMSIZE[1]}',  # size of the input video
        '-r', str(FPS),  # frames per second
        '-i', '-',  # input comes from a pipe

        # Add dummy audio
        '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo', # dummy audio input

        # Video settings
        '-c:v', 'libx264',  # video codec
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-b:v', '250k',  # Adjust bitrate as needed for better quality
        '-maxrate', '300k',
        '-bufsize', '600k',
        '-pix_fmt', 'yuv420p', #standard pixel format for compatibility
        '-profile:v', 'baseline', #Baseline for better compatibility with mobile devices

        # Audio Setting
        
        '-c:a', 'aac',  # Codec audio
        '-b:a', '128k',  # Bitrate audio
        '-ac', '2',
        '-f', 'flv',  # format for RTMP
        f'rtmp://{stream_ip}:1935/markaswalet-live/{stream_key}'  # RTMP server URL
    ]

    # Start the camera
    print('\033c')
    print(f'====================== START STREAM  =============================')
    print(f'Start Streaming with stream id = {stream_key}')
    print(f'rtmp://{stream_ip}:1935/markaswalet-live/{stream_key}')
    print(f'====================== START CAMERA  =============================')
    picam2.start()
    print(f'====================== START SENDING =============================')
    # setups ir led for night and day
    
    chip = gpiod.Chip('gpiochip4')
    led_line = chip.get_line(IRLED)
    led_line.request(consumer="LED",
    type=gpiod.LINE_REQ_DIR_OUT,
    flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP)
    # Start ffmpeg subprocess
    ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE)
    # take picture and save it
    frame = picam2.capture_array()
    if frame is None:
        print('Outer Image Capture Error')
    frame_bgr = np.asarray(frame[:, :, 0:3], dtype=np.uint8)
    frame_bgr = cv2.flip(frame_bgr, -1)
    print('Outer Image Capture Success')
    write_image(frame_bgr)
    try:
        
        while True:
            # Turn on relay
            relay_on_time_between(LED_LINE=led_line)
            # Capture video frame
            frame = picam2.capture_array()
            if frame is None:
                break

            # frame_bgr = frame[:, :, 0:3]  # Extract RGB channels from XRGB8888

            # Convert from numpy array to OpenCV image format
            frame_bgr = np.asarray(frame[:, :, 0:3], dtype=np.uint8)
            frame_bgr = cv2.flip(frame_bgr, -1)
            write_image(frame_bgr)
            
            # yolo
            # new_frame_bgr, count = detector.detect_and_count_birds(frame, confidence=0.65)

            # haarcascade
            new_frame_bgr, count = get_faces(frame_bgr)
            # Get the current time
            current_time = datetime.now().strftime("%H:%M:%S")
            # Write it
            frame_rgb = cv2.cvtColor(new_frame_bgr, cv2.COLOR_BGR2RGB)
            cv2.putText(frame_rgb, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) # red
            try:
                ffmpeg.stdin.write(frame_rgb.tobytes())
            except Exception as e:
                print(f"Error writing to ffmpeg stdin: {e}")
                break


    except KeyboardInterrupt:
        print("Streaming stopped by user")

    finally:
        # Clean up
        print(f'====================== STREAM ERROR        =============================')
        ffmpeg.stdin.close()
        ffmpeg.wait()
        picam2.stop()
        led_line.release()
    print(f'====================== RE-STARTING PROCESS =============================\n\n\n')
    time.sleep(5)
    
    
