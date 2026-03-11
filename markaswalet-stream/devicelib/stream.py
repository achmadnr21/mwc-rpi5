# stream.py — streaming pipeline with structured logging
import numpy as np
import cv2
from picamera2 import Picamera2
from datetime import datetime, date
import zoneinfo
_WIB = zoneinfo.ZoneInfo('Asia/Jakarta')
import time
import subprocess
import os
import logging
import traceback

# Import SwiftletCounter for frame processing
from devicelib.swiftlet_counter import SwiftletCounter
from devicelib.device import Device

import gpiod

logger = logging.getLogger(__name__)

# How often (seconds) the RPi polls the API for updated config
CONFIG_POLL_INTERVAL = 60
# How often (seconds) the RPi reports accumulated bird counts to the API
BIRD_COUNT_REPORT_INTERVAL = 30  # 30 seconds for near-realtime statistics

IRLED = 16

# Directory that contains this file — used for resolving relative assets
_STREAM_DIR = os.path.dirname(os.path.abspath(__file__))

# start time default is 17:00 and turn it off next day at 5:00
def relay_on_time_between(LED_LINE=None):
    if LED_LINE is None:
        return
    start_time = 17
    end_time = 5
    current_time = datetime.now(_WIB).hour
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

# Initialize PiCamera2
logger.debug('[CAMERA] Initialising PiCamera2…')
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": IMSIZE}))
logger.debug('[CAMERA] PiCamera2 configured')


# Initialize SwiftletCounter for streaming (frame-by-frame)
class StreamingSwiftletCounter(SwiftletCounter):
    def __init__(self, config_path=None, device_name="RBW Lantai 1", yolo_model_path=None):
        # Resolve config path relative to the markaswalet-stream directory
        if config_path is None:
            # Walk up one level from devicelib/ to markaswalet-stream/
            _parent = os.path.dirname(_STREAM_DIR)
            config_path = os.path.join(_parent, 'config.json')
            logger.debug(f'[CONFIG] Resolved config path: {config_path}')

        if not os.path.isfile(config_path):
            logger.warning(f'[CONFIG] config.json NOT FOUND at: {config_path}')
            logger.warning('[CONFIG] Falling back to built-in defaults')
        else:
            logger.info(f'[CONFIG] config.json found at: {config_path}')

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


def _select_video_encoder() -> str:
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
            logger.info('[FFMPEG] Hardware encoder h264_v4l2m2m available')
            return 'h264_v4l2m2m'
    except Exception as e:
        logger.warning(f'[FFMPEG] Could not query encoders: {e}')
    logger.info('[FFMPEG] Falling back to software encoder libx264')
    return 'libx264'


def stream_process(
    stream_ip='103.193.179.252',
    stream_key='mwcdef',
    device_name='markaswalet-capture-device',
):
    logger.info(f'[STREAM] stream_process() called — key={stream_key}, ip={stream_ip}, device={device_name}')

    if stream_key == 'mwcdef' or stream_key is None:
        logger.error('[STREAM] Invalid stream key — exiting')
        exit(0)

    video_encoder = _select_video_encoder()
    logger.info(f'[STREAM] Using FFmpeg video encoder: {video_encoder}')

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

    rtmp_url = f'rtmp://{stream_ip}:1935/markaswalet-live/{stream_key}'
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
        '-f', 'flv',
        '-rtmp_live', 'live',
        rtmp_url
    ]

    logger.info(f'[STREAM] RTMP target: {rtmp_url}')
    logger.info('[STREAM] Starting PiCamera2…')

    picam2.start()
    logger.info('[STREAM] PiCamera2 started — input source: PiCamera2')

    # Setup IR LED for night/day switching
    # Auto-detect the correct gpiochip — different RPi revisions use different numbers
    def _find_gpio_chip(required_lines: int) -> 'gpiod.Chip | None':
        """Scan /dev/gpiochipN (0-9) and return the first chip that has
        at least `required_lines` GPIO lines available."""
        for idx in range(10):
            chip_path = f'gpiochip{idx}'
            dev_path  = f'/dev/{chip_path}'
            if not os.path.exists(dev_path):
                continue
            try:
                c = gpiod.Chip(chip_path)
                if c.num_lines() > required_lines:
                    logger.info(f'[GPIO] Using {chip_path} ({c.num_lines()} lines) for IR LED')
                    return c
                c.close()
            except Exception as _e:
                logger.debug(f'[GPIO] {chip_path} probe failed: {_e}')
        return None

    logger.debug(f'[GPIO] Searching for gpiochip with >{IRLED} lines…')
    led_line = None
    try:
        chip = _find_gpio_chip(IRLED)
        if chip is None:
            logger.warning('[GPIO] No suitable gpiochip found — IR LED disabled, stream continues')
        else:
            led_line = chip.get_line(IRLED)
            led_line.request(
                consumer="LED",
                type=gpiod.LINE_REQ_DIR_OUT,
                flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP
            )
            logger.info(f'[GPIO] IR LED line {IRLED} acquired on {chip.name()}')
    except Exception as e:
        logger.error(f'[GPIO] Failed to set up IR LED: {e} — stream continues without IR LED')
        logger.debug(traceback.format_exc())
        led_line = None

    # Start ffmpeg subprocess
    logger.info('[FFMPEG] Launching FFmpeg subprocess…')
    try:
        ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, bufsize=0)
        logger.info(f'[FFMPEG] Process PID: {ffmpeg.pid}')
    except Exception as e:
        logger.critical(f'[FFMPEG] Failed to start FFmpeg: {e}')
        logger.critical(traceback.format_exc())
        picam2.stop()
        if led_line:
            led_line.release()
        return

    # Auto-detect ONNX model in parent directory of this file
    _here = os.path.dirname(os.path.abspath(__file__))
    _onnx_path = os.path.join(os.path.dirname(_here), 'swiftlet_yolov8n.onnx')
    if os.path.isfile(_onnx_path):
        yolo_path = _onnx_path
        logger.info(f'[YOLO] ONNX model found: {yolo_path}')
    else:
        yolo_path = None
        logger.warning(f'[YOLO] ONNX model NOT found at: {_onnx_path} — using classical CV')

    # Initialize SwiftletCounter for streaming
    logger.info('[COUNTER] Initialising StreamingSwiftletCounter…')
    try:
        swiftlet_counter = StreamingSwiftletCounter(device_name=device_name, yolo_model_path=yolo_path)
        logger.info('[COUNTER] StreamingSwiftletCounter ready')
    except Exception as e:
        logger.critical(f'[COUNTER] Failed to initialise SwiftletCounter: {e}')
        logger.critical(traceback.format_exc())
        ffmpeg.stdin.close()
        ffmpeg.wait()
        picam2.stop()
        if led_line:
            led_line.release()
        return

    # Device object for config polling
    _device_cfg = Device()
    _last_config_poll = 0.0
    _last_count_report = 0.0
    _current_date = datetime.now(_WIB).date()  # Track WIB date for daily counter reset

    # Restore today's accumulated counts so the counter does not reset to 0 on restart
    logger.info('[RESTORE] Fetching today\'s counts from API…')
    try:
        _today_saved = _device_cfg.get_today_count()
        _last_reported_in  = _today_saved['total_in']
        _last_reported_out = _today_saved['total_out']
        swiftlet_counter.cross_in_to_out  = _last_reported_in
        swiftlet_counter.cross_out_to_in  = _last_reported_out
        swiftlet_counter.crossing_count   = _today_saved['crossing_count']
        logger.info(f'[RESTORE] crossing_count={swiftlet_counter.crossing_count} total_in={_last_reported_in} total_out={_last_reported_out}')
    except Exception as e:
        logger.error(f'[RESTORE] Failed to restore today\'s counts: {e}')
        logger.debug(traceback.format_exc())
        _last_reported_in = 0
        _last_reported_out = 0

    # Initial test capture
    logger.debug('[CAMERA] Performing initial test capture…')
    try:
        frame = picam2.capture_array()
        if frame is None:
            logger.warning('[CAMERA] Initial test capture returned None')
        else:
            frame_bgr = np.asarray(frame[:, :, :3], dtype=np.uint8)
            frame_bgr = cv2.flip(frame_bgr, -1)
            logger.info('[CAMERA] Initial test capture successful')
    except Exception as e:
        logger.error(f'[CAMERA] Initial test capture failed: {e}')
        logger.debug(traceback.format_exc())

    frame_counter = 0
    logger.info('[STREAM] Entering main capture loop…')
    try:
        while True:
            # Daily counter reset at midnight (WIB)
            _now = time.time()
            _today = datetime.now(_WIB).date()
            if _today != _current_date:
                _current_date = _today
                swiftlet_counter.cross_in_to_out  = 0
                swiftlet_counter.cross_out_to_in  = 0
                swiftlet_counter.crossing_count   = 0
                _last_reported_in  = 0
                _last_reported_out = 0
                logger.info(f'[DAILY_RESET] Counters reset for new day: {_today}')

            # Poll API for updated config every CONFIG_POLL_INTERVAL seconds
            if _now - _last_config_poll >= CONFIG_POLL_INTERVAL:
                _last_config_poll = _now
                try:
                    remote_config = _device_cfg.get_config()
                    if remote_config is not None:
                        swiftlet_counter.reload_config_from_dict(remote_config)
                        logger.debug('[CONFIG_POLL] Config refreshed from API')
                    else:
                        logger.debug('[CONFIG_POLL] No remote config available')
                except Exception as _e:
                    logger.error(f'[CONFIG_POLL] Error fetching config: {_e}')
                    logger.debug(traceback.format_exc())

            # Report bird count delta every BIRD_COUNT_REPORT_INTERVAL seconds
            if _now - _last_count_report >= BIRD_COUNT_REPORT_INTERVAL:
                _last_count_report = _now
                current_in  = swiftlet_counter.cross_in_to_out
                current_out = swiftlet_counter.cross_out_to_in
                delta_in    = current_in  - _last_reported_in
                delta_out   = current_out - _last_reported_out
                try:
                    success = _device_cfg.report_bird_count(delta_in, delta_out, swiftlet_counter.crossing_count)
                    if success:
                        if delta_in > 0 or delta_out > 0:
                            _last_reported_in  = current_in
                            _last_reported_out = current_out
                            logger.info(f'[BIRD_COUNT] Reported delta_in={delta_in} delta_out={delta_out} crossing={swiftlet_counter.crossing_count}')
                        else:
                            logger.debug(f'[BIRD_COUNT] Updated crossing={swiftlet_counter.crossing_count} (no delta)')
                    else:
                        logger.warning('[BIRD_COUNT] API report returned failure')
                except Exception as _e:
                    logger.error(f'[BIRD_COUNT] Error reporting count: {_e}')
                    logger.debug(traceback.format_exc())

            # Turn on relay (IR LED) based on time-of-day
            if led_line:
                relay_on_time_between(LED_LINE=led_line)

            # Capture frame from PiCamera2
            # XRGB8888: channels are [B, G, R, X] on RPi (little-endian) → take first 3 = BGR
            try:
                frame = picam2.capture_array()
            except Exception as e:
                logger.error(f'[CAMERA] capture_array() failed: {e}')
                logger.debug(traceback.format_exc())
                break

            if frame is None:
                logger.error('[CAMERA] capture_array() returned None — stopping stream')
                break

            frame_bgr = np.asarray(frame[:, :, :3], dtype=np.uint8)
            frame_bgr = cv2.flip(frame_bgr, -1)  # Rotate 180° for upside-down mounted camera

            # Process frame with SwiftletCounter (detect, annotate)
            try:
                processed_frame = swiftlet_counter.process_frame(frame_bgr)
            except Exception as e:
                logger.error(f'[COUNTER] process_frame() failed on frame #{frame_counter}: {e}')
                logger.debug(traceback.format_exc())
                processed_frame = frame_bgr  # pass raw frame on error

            # Timestamp overlay
            current_time_str = datetime.now(_WIB).strftime("%H:%M:%S")
            cv2.putText(processed_frame, current_time_str, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            try:
                ffmpeg.stdin.write(processed_frame.tobytes())
            except Exception as e:
                logger.error(f'[FFMPEG] stdin write failed on frame #{frame_counter}: {e}')
                logger.debug(traceback.format_exc())
                break

            frame_counter += 1
            if frame_counter % (FPS * 60) == 0:  # log every ~1 minute at runtime
                mins = frame_counter // (FPS * 60)
                logger.info(f'[STREAM] Running {mins} min — in={swiftlet_counter.cross_in_to_out} out={swiftlet_counter.cross_out_to_in} crossing={swiftlet_counter.crossing_count}')

    except KeyboardInterrupt:
        logger.info('[STREAM] Streaming stopped by user (KeyboardInterrupt)')

    except Exception as e:
        logger.critical(f'[STREAM] Unexpected exception in capture loop: {e}')
        logger.critical(traceback.format_exc())

    finally:
        logger.info('[STREAM] Cleaning up resources…')
        try:
            ffmpeg.stdin.close()
            ffmpeg.wait()
            logger.info('[FFMPEG] FFmpeg process terminated')
        except Exception as e:
            logger.warning(f'[FFMPEG] Error during cleanup: {e}')
        try:
            picam2.stop()
            logger.info('[CAMERA] PiCamera2 stopped')
        except Exception as e:
            logger.warning(f'[CAMERA] Error stopping PiCamera2: {e}')
        if led_line:
            try:
                led_line.release()
                logger.info('[GPIO] IR LED line released')
            except Exception as e:
                logger.warning(f'[GPIO] Error releasing LED line: {e}')

    logger.info('[STREAM] stream_process() finished — will restart in 5 s')
    time.sleep(5)
