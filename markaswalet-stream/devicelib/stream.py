# in this file, we define the Stream procedure to perform the streaming process
import numpy as np
import cv2
from picamera2 import Picamera2
from datetime import datetime
import time
import subprocess
import os
# import devicelib.detector as detector

# Import SwiftletCounter for frame processing
from markaswalet_stream.devicelib.swiftlet_counter import SwiftletCounter

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

IMSIZE = (640, 480)
FPS = 15  # Adjust to your desired frame rate

# Initialize the PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": IMSIZE}))

# Initialize SwiftletCounter for streaming (frame-by-frame)
class StreamingSwiftletCounter(SwiftletCounter):
    def __init__(self, config_path="config.json"):
        # Dummy paths, not used in streaming mode
        super().__init__(input_video_path=None, output_video_path=None, config_path=config_path)
        self.streaming_mode = True
        self.fps = FPS
        self.width = IMSIZE[0]
        self.height = IMSIZE[1]
        self.out = None  # No video writer
        self.cap = None  # No video capture

    def process_frame(self, frame):
        # Simulate the main pipeline for a single frame
        # Preprocess, detect, update trackers, annotate
        fg_mask = self.bg_subtractor.apply(frame)
        mask = self._preprocess_mask(fg_mask)
        detections = self.detect_birds(frame, mask)
        detections = self._apply_temporal_consistency(detections)
        self.update_trackers(frame, detections)
        annotated = self.draw_annotations(frame)
        self.frame_count += 1
        self.detection_history.append(len(detections))
        return annotated


def stream_process(stream_ip = '103.193.179.252' ,stream_key='mwcdef'):

    if stream_key == 'mwcdef' or stream_key is None:
        print('Exiting from the program')
        exit(0)
    

    command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', 
        '-s', f'{IMSIZE[0]}x{IMSIZE[1]}', 
        '-r', str(FPS), 
        '-i', '-', 

        '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo', 

        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-b:v', '250k',
        '-maxrate', '300k',
        '-bufsize', '600k',
        '-pix_fmt', 'yuv420p',
        '-profile:v', 'baseline',

        '-c:a', 'aac',
        '-b:a', '128k',
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
    # Initialize SwiftletCounter for streaming
    swiftlet_counter = StreamingSwiftletCounter()
    frame = picam2.capture_array()
    if frame is None:
        print('Outer Image Capture Error')
    else:
        frame_bgr = np.asarray(frame[:, :, 0:3], dtype=np.uint8)
        frame_bgr = cv2.flip(frame_bgr, -1)
        print('Outer Image Capture Success')
    try:
        while True:
            # Turn on relay
            relay_on_time_between(LED_LINE=led_line)
            # Capture video frame
            frame = picam2.capture_array()
            if frame is None:
                break

            # Convert from numpy array to OpenCV image format
            frame_bgr = np.asarray(frame[:, :, 0:3], dtype=np.uint8)
            frame_bgr = cv2.flip(frame_bgr, -1)

            # Process frame with SwiftletCounter (detect, annotate)
            processed_frame = swiftlet_counter.process_frame(frame_bgr)

            # Add timestamp overlay (optional, can be moved to draw_annotations)
            current_time = datetime.now().strftime("%H:%M:%S")
            cv2.putText(processed_frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) # red

            # Convert to RGB for ffmpeg
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
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
    
    
