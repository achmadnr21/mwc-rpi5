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
from devicelib.device import Device

# How often (seconds) the RPi polls the API for updated config
CONFIG_POLL_INTERVAL = 60

TEST_VIDEO_PATH = 'test_video.mp4'  # For testing on non-Raspberry Pi environments
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
GOP_SIZE = int(FPS * 1.5)  # 1.5-second GOP, better quality at same latency class
VIDEO_BITRATE = '280k'
VIDEO_MAXRATE = '320k'
VIDEO_BUFSIZE = '640k'

# Initialize the PiCamera2
# picam2 = Picamera2()  # Raspberry Pi specific
# picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": IMSIZE}))

# Initialize SwiftletCounter for streaming (frame-by-frame)
class StreamingSwiftletCounter(SwiftletCounter):
    def __init__(self, config_path="config.json", device_name="RBW Lantai 1", yolo_model_path=None):
        super().__init__(
            input_video_path=None,
            output_video_path=None,
            config_path=config_path,
            streaming_mode=True,
            device_name=device_name,
            yolo_model_path=yolo_model_path,
        )
        self.fps = FPS
        self.width = IMSIZE[0]
        self.height = IMSIZE[1]

    def process_frame(self, frame):
        # Apply color preprocessing (no-op when all params are at default)
        if self._color_preprocessing_active():
            frame = self._apply_color_preprocessing(frame)
        # Process single frame: detect, track, annotate
        fg_mask = self.bg_subtractor.apply(frame)
        mask = self._preprocess_mask(fg_mask)
        detections = self.detect_birds(frame, mask)
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


def stream_process(
    stream_ip='103.193.179.252',
    stream_key='mwcdef',
    device_name='markaswalet-capture-device',
    use_test_video=False,
    test_video_path=TEST_VIDEO_PATH
):

    if stream_key == 'mwcdef' or stream_key is None:
        print('Exiting from the program')
        exit(0)
    

    video_encoder = _select_video_encoder()
    print(f'Using FFmpeg video encoder: {video_encoder}')

    if video_encoder == 'h264_v4l2m2m':
        video_encode_args = [
            '-c:v', 'h264_v4l2m2m',
            '-b:v', VIDEO_BITRATE,
            '-maxrate', VIDEO_MAXRATE,
            '-bufsize', VIDEO_BUFSIZE,
            '-g', str(GOP_SIZE),
            '-keyint_min', str(GOP_SIZE),
            '-sc_threshold', '0',
            '-bf', '0',
            '-pix_fmt', 'yuv420p'
        ]
    else:
        video_encode_args = [
            '-c:v', 'libx264',
            '-preset', 'superfast',
            '-tune', 'zerolatency',
            '-b:v', VIDEO_BITRATE,
            '-maxrate', VIDEO_MAXRATE,
            '-bufsize', VIDEO_BUFSIZE,
            '-g', str(GOP_SIZE),
            '-keyint_min', str(GOP_SIZE),
            '-sc_threshold', '0',
            '-bf', '0',
            '-pix_fmt', 'yuv420p',
            '-profile:v', 'baseline',
            '-x264-params', f'keyint={GOP_SIZE}:min-keyint={GOP_SIZE}:scenecut=0:rc-lookahead=0'
        ]

    command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-y',
        '-fflags', 'nobuffer',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{IMSIZE[0]}x{IMSIZE[1]}', 
        '-r', str(FPS), 
        '-i', '-', 
    ] + video_encode_args + [
        '-flags', 'low_delay',
        '-an',
        '-flush_packets', '1',
        '-muxdelay', '0',
        '-muxpreload', '0',
        '-f', 'flv',  # format for RTMP
        '-rtmp_live', 'live',
        f'rtmp://{stream_ip}:1935/markaswalet-live/{stream_key}'  # RTMP server URL
    ]

    # Start the camera
    print('\033c')
    print(f'====================== START STREAM  =============================')
    print(f'Start Streaming with stream id = {stream_key}')
    print(f'rtmp://{stream_ip}:1935/markaswalet-live/{stream_key}')
    print(f'====================== START CAMERA  =============================')

    # picam2.start()  # Raspberry Pi specific
    if use_test_video:
        cap = cv2.VideoCapture(test_video_path)
        print(f'Input source: test video -> {test_video_path}')
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMSIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMSIZE[1])
        cap.set(cv2.CAP_PROP_FPS, FPS)
        print('Input source: local camera (cv2.VideoCapture(0))')

    if not cap.isOpened():
        if use_test_video:
            print(f'Cannot open test video: {test_video_path}')
        else:
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
    ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, bufsize=0)
    # take picture and save it
    # Auto-detect ONNX model in same directory as this file
    _here = os.path.dirname(os.path.abspath(__file__))
    _onnx_path = os.path.join(os.path.dirname(_here), 'swiftlet_yolov8n.onnx')
    yolo_path = _onnx_path if os.path.isfile(_onnx_path) else None

    # Initialize SwiftletCounter for streaming
    swiftlet_counter = StreamingSwiftletCounter(device_name=device_name, yolo_model_path=yolo_path)

    # Device object for config polling (reads id/password from local SQLite)
    _device_cfg = Device()
    _last_config_poll = 0.0

    ret, frame_bgr = cap.read()
    if not ret or frame_bgr is None:
        print('Outer Image Capture Error')
    else:
        frame_bgr = cv2.resize(frame_bgr, IMSIZE)
        frame_bgr = np.asarray(frame_bgr, dtype=np.uint8)
        print('Outer Image Capture Success')
    try:
        while True:
            # Poll API for updated config every CONFIG_POLL_INTERVAL seconds
            _now = time.time()
            if _now - _last_config_poll >= CONFIG_POLL_INTERVAL:
                _last_config_poll = _now
                try:
                    remote_config = _device_cfg.get_config()
                    if remote_config is not None:
                        swiftlet_counter.reload_config_from_dict(remote_config)
                except Exception as _e:
                    print(f'[CONFIG_POLL] Error fetching config: {_e}')

            # Turn on relay
            relay_on_time_between(LED_LINE=led_line)
            # Capture video frame
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                if use_test_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame_bgr = cap.read()
                    if not ret or frame_bgr is None:
                        print('Test video loop failed to read frame')
                        break
                else:
                    break

            frame_bgr = cv2.resize(frame_bgr, IMSIZE)
            frame_bgr = np.asarray(frame_bgr, dtype=np.uint8)

            # Process frame with SwiftletCounter (detect, annotate)
            processed_frame = swiftlet_counter.process_frame(frame_bgr)

            # Add timestamp overlay (optional, can be moved to draw_annotations)
            current_time = datetime.now().strftime("%H:%M:%S")
            cv2.putText(processed_frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) # red

            try:
                ffmpeg.stdin.write(processed_frame.tobytes())
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
    
    
