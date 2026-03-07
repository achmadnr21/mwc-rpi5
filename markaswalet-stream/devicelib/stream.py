# in this file, we define the Stream procedure to perform the streaming process
import numpy as np
import cv2
# from picamera2 import Picamera2  # Raspberry Pi specific
from datetime import datetime
import time
import subprocess
import os
# import devicelib.detector as detector

# Import SwiftletCounter for frame processing
from devicelib.swiftlet_counter import SwiftletCounter

# import raspberry pi pins to pull up digital pin for relay

# import gpiod  # Raspberry Pi specific

IRLED= 16
# start time default is 17:00 and turn it off next day at 5:00
def relay_on_time_between(LED_LINE = None):
    if LED_LINE is None:
        return
    start_time = 17
    end_time = 5
    current_time = datetime.now().hour
    if current_time >= start_time or current_time <= end_time:
        LED_LINE.set_value(0)
    else:
        LED_LINE.set_value(1)

IMSIZE = (640, 480)
FPS = 10  # Lighter streaming frame rate

# Initialize the PiCamera2
# picam2 = Picamera2()  # Raspberry Pi specific
# picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": IMSIZE}))

# Initialize SwiftletCounter for streaming (frame-by-frame)
class StreamingSwiftletCounter(SwiftletCounter):
    def __init__(self, config_path="config.json", device_name="RBW Lantai 1"):
        # Initialize parent with streaming mode flag
        super().__init__(input_video_path=None, output_video_path=None, config_path=config_path, streaming_mode=True, device_name=device_name)
        self.fps = FPS
        self.width = IMSIZE[0]
        self.height = IMSIZE[1]

    def process_frame(self, frame):
        # Process single frame: detect, track, annotate
        fg_mask = self.bg_subtractor.apply(frame)
        mask = self._preprocess_mask(fg_mask)
        detections = self.detect_birds(frame, mask)
        detections = self._apply_temporal_consistency(detections)
        self.update_trackers(frame, detections)
        annotated = self.draw_annotations(frame)
        self.frame_count += 1
        self.detection_history.append(len(detections))
        return annotated


def _select_video_encoder():
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True,
            text=True,
            check=False,
            timeout=5
        )
        encoders_text = f"{result.stdout}\n{result.stderr}"
        if 'h264_v4l2m2m' in encoders_text:
            return 'h264_v4l2m2m'
    except Exception:
        pass
    return 'libx264'


def stream_process(stream_ip = '103.193.179.252' ,stream_key='mwcdef', device_name = 'markaswalet-capture-device'):

    if stream_key == 'mwcdef' or stream_key is None:
        print('Exiting from the program')
        exit(0)
    

    video_encoder = _select_video_encoder()
    print(f'Using FFmpeg video encoder: {video_encoder}')

    if video_encoder == 'h264_v4l2m2m':
        video_encode_args = [
            '-c:v', 'h264_v4l2m2m',
            '-b:v', '150k',
            '-maxrate', '180k',
            '-bufsize', '360k',
            '-g', '30',
            '-keyint_min', '30',
            '-sc_threshold', '0',
            '-bf', '0',
            '-pix_fmt', 'yuv420p'
        ]
    else:
        video_encode_args = [
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-b:v', '150k',
            '-maxrate', '180k',
            '-bufsize', '360k',
            '-g', '30',
            '-keyint_min', '30',
            '-sc_threshold', '0',
            '-bf', '0',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'baseline'
        ]

    command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', 
        '-s', f'{IMSIZE[0]}x{IMSIZE[1]}', 
        '-r', str(FPS), 
        '-i', '-', 
    ] + video_encode_args + [
        '-an',
        '-f', 'flv',  # format for RTMP
        f'rtmp://{stream_ip}:1935/markaswalet-live/{stream_key}'  # RTMP server URL
    ]

    # Start the camera
    print('\033c')
    print(f'====================== START STREAM  =============================')
    print(f'Start Streaming with stream id = {stream_key}')
    print(f'rtmp://{stream_ip}:1935/markaswalet-live/{stream_key}')
    print(f'====================== START CAMERA  =============================')
    # picam2.start()  # Raspberry Pi specific
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMSIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMSIZE[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        print('Cannot open local camera (cv2.VideoCapture(0))')
        return
    print(f'====================== START SENDING =============================')
    # setups ir led for night and day
    
    # chip = gpiod.Chip('gpiochip4')  # Raspberry Pi specific
    # led_line = chip.get_line(IRLED)
    # led_line.request(consumer="LED",
    # type=gpiod.LINE_REQ_DIR_OUT,
    # flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP)
    led_line = None
    
    # Start ffmpeg subprocess
    ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE)
    # take picture and save it
    # Initialize SwiftletCounter for streaming
    swiftlet_counter = StreamingSwiftletCounter(device_name=device_name)
    ret, frame_bgr = cap.read()
    if not ret or frame_bgr is None:
        print('Outer Image Capture Error')
    else:
        frame_bgr = cv2.resize(frame_bgr, IMSIZE)
        frame_bgr = np.asarray(frame_bgr, dtype=np.uint8)
        print('Outer Image Capture Success')
    try:
        while True:
            # Turn on relay
            relay_on_time_between(LED_LINE=led_line)
            # Capture video frame
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            frame_bgr = cv2.resize(frame_bgr, IMSIZE)
            frame_bgr = np.asarray(frame_bgr, dtype=np.uint8)

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
        cap.release()
        # picam2.stop()  # Raspberry Pi specific
        # led_line.release()  # Raspberry Pi specific
    print(f'====================== RE-STARTING PROCESS =============================\n\n\n')
    time.sleep(5)
    
    
