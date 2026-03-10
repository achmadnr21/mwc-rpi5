import cv2
import numpy as np
import time
import json
import os
from pathlib import Path
from collections import deque
import math

class SwiftletCounter:
    def __init__(
        self,
        input_video_path=None,
        output_video_path=None,
        config_path="config.json",
        streaming_mode=False,
        device_name="RBW Lantai 1",
        center_box_width_ratio=None,
        center_box_height_ratio=None,
        yolo_model_path=None
    ):
        self.input_path = input_video_path
        self.output_path = output_video_path
        self.streaming_mode = streaming_mode
        self.device_name = device_name
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Video capture and properties (only for video file mode)
        if not streaming_mode and input_video_path:
            self.cap = cv2.VideoCapture(input_video_path)
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            self.cap = None
            self.fps = 15  # Default for streaming
            self.width = 640  # Default for streaming
            self.height = 480  # Default for streaming
        
        # Video writer for output (only for video file mode)
        if not streaming_mode and output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width, self.height))
        else:
            self.out = None
        
        # YOLO ONNX detector (optional — falls back to classical CV if not provided)
        self._yolo_net  = None   # custom YOLOv8n (swiftlet-trained)
        self._yolo11n_net = None  # YOLO11n COCO general bird detector
        self._use_contour_motion = False  # pure motion+contour model (no shape gates)
        self._prev_gray = None  # previous grayscale frame for frame differencing
        self._yolo_model_path = yolo_model_path  # stored for hot-reload
        # Derive YOLO11n path from same directory as the custom model
        _yolo_dir = os.path.dirname(os.path.abspath(yolo_model_path)) if yolo_model_path \
                    else os.path.dirname(os.path.abspath(__file__))
        self._yolo11n_model_path = os.path.join(_yolo_dir, 'yolo11n_bird.onnx')
        self._yolo_input_size     = int(self.config.get('yolo_input_size', 320))
        self._yolo_conf_threshold = self.config.get('yolo_conf_threshold', 0.40)
        self._yolo_nms_threshold  = self.config.get('yolo_nms_threshold', 0.45)
        use_yolo        = self.config.get('use_yolo', True)
        yolo_model_type = self.config.get('yolo_model_type', 'yolov8n')
        if use_yolo and yolo_model_type == 'yolov8n' and yolo_model_path and os.path.isfile(yolo_model_path):
            self._yolo_net = cv2.dnn.readNetFromONNX(yolo_model_path)
            print(f'[YOLO] Loaded YOLOv8n (swiftlet): {yolo_model_path}')
        elif use_yolo and yolo_model_type == 'yolo11n_coco' and os.path.isfile(self._yolo11n_model_path):
            self._yolo11n_net = cv2.dnn.readNetFromONNX(self._yolo11n_model_path)
            print(f'[YOLO11n] Loaded COCO bird model: {self._yolo11n_model_path}')
        elif yolo_model_type == 'contour_motion':
            self._use_contour_motion = True
            print('[CV] Contour Motion model active — motion+blob detection, no shape gates')
        elif not use_yolo:
            print('[YOLO] use_yolo=False in config — using classical CV detection')
        else:
            print('[YOLO] No ONNX model found — using classical CV detection')

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Tracking parameters
        self.trackers = []
        self.bird_id_counter = 0
        self.bird_count = 0
        self.crossing_count = 0
        self.cross_in_to_out = 0
        self.cross_out_to_in = 0
        self.max_birds_per_frame = 20  # Prevent tracker explosion
        config_center_width = self.config.get('center_box_width_ratio', 0.35)
        config_center_height = self.config.get('center_box_height_ratio', 0.35)
        self.center_box_width_ratio = center_box_width_ratio if center_box_width_ratio is not None else config_center_width
        self.center_box_height_ratio = center_box_height_ratio if center_box_height_ratio is not None else config_center_height
        self.center_box_width_ratio = max(0.05, min(0.95, self.center_box_width_ratio))
        self.center_box_height_ratio = max(0.05, min(0.95, self.center_box_height_ratio))
        self.center_gate_margin_ratio = self.config.get('center_gate_margin_ratio', 0.03)
        self.center_gate_margin_ratio = max(0.0, min(0.2, self.center_gate_margin_ratio))
        self.center_box = self._get_center_box(self.width, self.height)
        
        # Detection parameters - back to working values
        self.min_contour_area = self.config.get('min_contour_area', 50)
        self.max_contour_area = self.config.get('max_contour_area', 500)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        # Contour Motion model params (tuned for tiny swiftlet blobs)
        self.cm_min_area       = self.config.get('cm_min_area', 12)
        self.cm_max_area       = self.config.get('cm_max_area', 700)
        self.cm_diff_threshold = self.config.get('cm_diff_threshold', 12)
        self.display_confidence_threshold = self.config.get('display_confidence_threshold', self.confidence_threshold)
        self.display_confidence_threshold = max(0.0, min(1.0, self.display_confidence_threshold))
        self.tracker_max_distance = self.config.get('tracker_max_distance', 70)
        self.tracker_max_lost_frames = self.config.get('tracker_max_lost_frames', 15)
        self.tracker_distance_growth = self.config.get('tracker_distance_growth', 0.35)
        self.tracker_max_distance_cap = self.config.get('tracker_max_distance_cap', 220)
        self.tracker_velocity_damping = self.config.get('tracker_velocity_damping', 0.85)
        self.tracker_min_confidence_keep = self.config.get('tracker_min_confidence_keep', 0.25)
        self.use_tracker_prediction = self.config.get('use_tracker_prediction', False)
        self.counting_confidence_threshold = self.config.get('counting_confidence_threshold', 0.65)
        self.counting_min_displacement_px = self.config.get('counting_min_displacement_px', 12)
        self.tracker_count_cooldown_seconds = self.config.get('tracker_count_cooldown_seconds', 1.2)
        self.global_count_min_interval_seconds = self.config.get('global_count_min_interval_seconds', 0.15)
        self.count_allow_prediction = self.config.get('count_allow_prediction', True)
        self.count_max_lost_frames = self.config.get('count_max_lost_frames', 2)
        self.count_recent_detection_window = self.config.get('count_recent_detection_window', 6)
        self.count_prediction_min_speed = self.config.get('count_prediction_min_speed', 4.5)
        self.count_relaxed_confidence_threshold = self.config.get(
            'count_relaxed_confidence_threshold',
            max(0.35, self.counting_confidence_threshold - 0.15)
        )
        self._last_count_time = 0.0
        self.bbox_smooth_alpha = self.config.get('bbox_smooth_alpha', 0.65)
        self.max_bbox_size_change = self.config.get('max_bbox_size_change', 2.2)
        
        # Morphological kernels
        kernel_size = self.config.get('morphology_kernel_size', 3)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        close_k = self.config.get('morph_close_kernel_size', 7)
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

        # Shape validation thresholds
        self.shape_min_solidity    = self.config.get('shape_min_solidity', 0.30)
        self.shape_min_compactness = self.config.get('shape_min_compactness', 0.08)
        self.shape_min_extent      = self.config.get('shape_min_extent', 0.18)
        self.shape_aspect_min      = self.config.get('shape_aspect_min', 0.25)
        self.shape_aspect_max      = self.config.get('shape_aspect_max', 4.5)

        # Confidence weights (5-feature)
        self.conf_w_area       = self.config.get('confidence_weight_area', 0.25)
        self.conf_w_aspect     = self.config.get('confidence_weight_aspect', 0.20)
        self.conf_w_solidity   = self.config.get('confidence_weight_solidity', 0.20)
        self.conf_w_compactness= self.config.get('confidence_weight_compactness', 0.15)
        self.conf_w_darkness   = self.config.get('confidence_weight_darkness', 0.20)

        # NMS merge params
        self.nms_iou_threshold  = self.config.get('nms_iou_threshold', 0.10)
        self.nms_merge_distance = self.config.get('nms_merge_distance', 20)

        # Kalman filter flag
        self.use_kalman_filter = self.config.get('use_kalman_filter', True)
        self.kalman_pn_pos = float(self.config.get('kalman_process_noise_pos', 1.0))
        self.kalman_pn_vel = float(self.config.get('kalman_process_noise_vel', 4.0))
        self.kalman_mn     = float(self.config.get('kalman_measurement_noise', 4.0))

        # Pre-tracking temporal vote pool
        self.pending_vote_frames  = self.config.get('pending_vote_frames', 2)
        self.pending_vote_window  = self.config.get('pending_vote_window', 4)
        self.pending_max_distance = self.config.get('pending_max_distance', 25)
        self._pending_pool = []

        # Color preprocessing parameters
        self.color_brightness = float(self.config.get('color_brightness', 0))
        self.color_contrast   = float(self.config.get('color_contrast', 0))
        self.color_saturation = float(self.config.get('color_saturation', 0))
        self.color_hue        = float(self.config.get('color_hue', 0))
        self.color_gamma      = float(self.config.get('color_gamma', 1.0))
        self.color_use_clahe  = bool(self.config.get('color_use_clahe', False))
        self.color_clahe_clip = float(self.config.get('color_clahe_clip', 2.0))

        # Statistics
        self.frame_count = 0
        self.detection_history = deque(maxlen=100)

        # FPS tracking — rolling window of the last 30 frame timestamps
        self._fps_times = deque(maxlen=30)
        
    def detect_birds(self, frame, mask):
        """Bird detection — routes to active model: YOLOv8n, YOLO11n COCO, Contour Motion, or classical CV."""
        if self._yolo_net is not None:
            return self._detect_yolo(frame)
        if self._yolo11n_net is not None:
            return self._detect_yolo11n_coco(frame)
        if self._use_contour_motion:
            return self._detect_contour_motion(frame, mask)
        return self._detect_motion_birds(frame, mask)

    # ── Color preprocessing ─────────────────────────────────────────────────

    def _color_preprocessing_active(self) -> bool:
        return (self.color_brightness != 0 or self.color_contrast != 0 or
                self.color_saturation != 0 or self.color_hue != 0 or
                self.color_gamma != 1.0 or self.color_use_clahe)

    def _apply_color_preprocessing(self, frame):
        """Return a color-corrected copy of frame based on current config params."""
        # Brightness + Contrast via convertScaleAbs: alpha=(1+c/100), beta=b
        b, c = self.color_brightness, self.color_contrast
        if b != 0 or c != 0:
            alpha = max(0.0, 1.0 + c / 100.0)
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=b)

        # Saturation + Hue shift via HSV
        if self.color_saturation != 0 or self.color_hue != 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            if self.color_hue != 0:
                hsv[:, :, 0] = (hsv[:, :, 0] + self.color_hue / 2.0) % 180
            if self.color_saturation != 0:
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + self.color_saturation / 100.0), 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Gamma correction via LUT
        if self.color_gamma != 1.0:
            inv_gamma = 1.0 / max(0.01, self.color_gamma)
            lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
            frame = cv2.LUT(frame, lut)

        # CLAHE (adaptive histogram equalisation on L channel)
        if self.color_use_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.color_clahe_clip, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return frame

    # ── Hot-reload ───────────────────────────────────────────────────────────

    def reload_config_from_dict(self, config_dict: dict):
        """Re-apply all tunable params from a dict (e.g. fetched from API).
        Counters and trackers are preserved — no data is lost."""
        self.config = config_dict
        self._yolo_conf_threshold = self.config.get('yolo_conf_threshold', 0.40)
        self._yolo_nms_threshold  = self.config.get('yolo_nms_threshold', 0.45)
        self._yolo_input_size     = int(self.config.get('yolo_input_size', 320))
        self.color_brightness     = float(self.config.get('color_brightness', 0))
        self.color_contrast       = float(self.config.get('color_contrast', 0))
        self.color_saturation     = float(self.config.get('color_saturation', 0))
        self.color_hue            = float(self.config.get('color_hue', 0))
        self.color_gamma          = float(self.config.get('color_gamma', 1.0))
        self.color_use_clahe      = bool(self.config.get('color_use_clahe', False))
        self.color_clahe_clip     = float(self.config.get('color_clahe_clip', 2.0))
        self.min_contour_area     = self.config.get('min_contour_area', 50)
        self.max_contour_area     = self.config.get('max_contour_area', 500)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.display_confidence_threshold = max(0.0, min(1.0, self.config.get(
            'display_confidence_threshold', self.confidence_threshold)))
        self.tracker_max_distance         = self.config.get('tracker_max_distance', 70)
        self.tracker_max_lost_frames      = self.config.get('tracker_max_lost_frames', 15)
        self.tracker_distance_growth      = self.config.get('tracker_distance_growth', 0.35)
        self.tracker_max_distance_cap     = self.config.get('tracker_max_distance_cap', 220)
        self.tracker_velocity_damping     = self.config.get('tracker_velocity_damping', 0.85)
        self.tracker_min_confidence_keep  = self.config.get('tracker_min_confidence_keep', 0.25)
        self.use_tracker_prediction       = self.config.get('use_tracker_prediction', False)
        self.counting_confidence_threshold = self.config.get('counting_confidence_threshold', 0.65)
        self.counting_min_displacement_px  = self.config.get('counting_min_displacement_px', 12)
        self.tracker_count_cooldown_seconds    = self.config.get('tracker_count_cooldown_seconds', 1.2)
        self.global_count_min_interval_seconds = self.config.get('global_count_min_interval_seconds', 0.15)
        self.count_allow_prediction        = self.config.get('count_allow_prediction', True)
        self.count_max_lost_frames         = self.config.get('count_max_lost_frames', 2)
        self.count_recent_detection_window = self.config.get('count_recent_detection_window', 6)
        self.count_prediction_min_speed    = self.config.get('count_prediction_min_speed', 4.5)
        self.count_relaxed_confidence_threshold = self.config.get(
            'count_relaxed_confidence_threshold',
            max(0.35, self.counting_confidence_threshold - 0.15))
        self.bbox_smooth_alpha    = self.config.get('bbox_smooth_alpha', 0.65)
        self.max_bbox_size_change = self.config.get('max_bbox_size_change', 2.2)
        kernel_size = self.config.get('morphology_kernel_size', 3)
        self.morph_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        close_k = self.config.get('morph_close_kernel_size', 7)
        self.close_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        self.shape_min_solidity    = self.config.get('shape_min_solidity', 0.30)
        self.shape_min_compactness = self.config.get('shape_min_compactness', 0.08)
        self.shape_min_extent      = self.config.get('shape_min_extent', 0.18)
        self.shape_aspect_min      = self.config.get('shape_aspect_min', 0.25)
        self.shape_aspect_max      = self.config.get('shape_aspect_max', 4.5)
        self.conf_w_area        = self.config.get('confidence_weight_area', 0.25)
        self.conf_w_aspect      = self.config.get('confidence_weight_aspect', 0.20)
        self.conf_w_solidity    = self.config.get('confidence_weight_solidity', 0.20)
        self.conf_w_compactness = self.config.get('confidence_weight_compactness', 0.15)
        self.conf_w_darkness    = self.config.get('confidence_weight_darkness', 0.20)
        self.nms_iou_threshold  = self.config.get('nms_iou_threshold', 0.10)
        self.nms_merge_distance = self.config.get('nms_merge_distance', 20)
        self.cm_min_area        = self.config.get('cm_min_area', 12)
        self.cm_max_area        = self.config.get('cm_max_area', 700)
        self.cm_diff_threshold  = self.config.get('cm_diff_threshold', 12)
        self.use_kalman_filter  = self.config.get('use_kalman_filter', True)
        self.kalman_pn_pos      = float(self.config.get('kalman_process_noise_pos', 1.0))
        self.kalman_pn_vel      = float(self.config.get('kalman_process_noise_vel', 4.0))
        self.kalman_mn          = float(self.config.get('kalman_measurement_noise', 4.0))
        self.pending_vote_frames  = self.config.get('pending_vote_frames', 2)
        self.pending_vote_window  = self.config.get('pending_vote_window', 4)
        self.pending_max_distance = self.config.get('pending_max_distance', 25)
        self.center_gate_margin_ratio = max(0.0, min(0.2, self.config.get('center_gate_margin_ratio', 0.03)))
        # Hot-reload model selection
        use_yolo        = self.config.get('use_yolo', True)
        yolo_model_type = self.config.get('yolo_model_type', 'yolov8n')
        if use_yolo and yolo_model_type == 'yolov8n':
            if self._yolo_net is None and self._yolo_model_path and os.path.isfile(self._yolo_model_path):
                self._yolo_net = cv2.dnn.readNetFromONNX(self._yolo_model_path)
                print(f'[CONFIG] YOLOv8n (swiftlet) loaded')
            if self._yolo11n_net is not None:
                self._yolo11n_net = None
                print('[CONFIG] YOLO11n unloaded')
            self._use_contour_motion = False
        elif use_yolo and yolo_model_type == 'yolo11n_coco':
            if self._yolo11n_net is None and os.path.isfile(self._yolo11n_model_path):
                self._yolo11n_net = cv2.dnn.readNetFromONNX(self._yolo11n_model_path)
                print(f'[CONFIG] YOLO11n COCO bird model loaded')
            if self._yolo_net is not None:
                self._yolo_net = None
                print('[CONFIG] YOLOv8n unloaded')
            self._use_contour_motion = False
        elif yolo_model_type == 'contour_motion':
            if self._yolo_net is not None:
                self._yolo_net = None
                print('[CONFIG] YOLOv8n unloaded → Contour Motion')
            if self._yolo11n_net is not None:
                self._yolo11n_net = None
                print('[CONFIG] YOLO11n unloaded → Contour Motion')
            if not self._use_contour_motion:
                self._use_contour_motion = True
                self._prev_gray = None  # reset frame diff on model switch
                print('[CONFIG] Contour Motion model activated')
        else:  # contour (classical shape-based)
            if self._yolo_net is not None:
                self._yolo_net = None
                print('[CONFIG] YOLOv8n unloaded → classical CV')
            if self._yolo11n_net is not None:
                self._yolo11n_net = None
                print('[CONFIG] YOLO11n unloaded → classical CV')
            self._use_contour_motion = False
        print('[CONFIG] Config hot-reloaded from dict')

    def _detect_yolo(self, frame):
        """YOLOv8n ONNX inference via cv2.dnn.

        Input: BGR frame at any resolution — resized internally to 320×320.
        Output: same detection dict format as _detect_motion_birds so the
        tracker pipeline is completely unchanged.
        """
        h, w = frame.shape[:2]
        sz = self._yolo_input_size
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (sz, sz), swapRB=True, crop=False)
        self._yolo_net.setInput(blob)
        raw = self._yolo_net.forward()          # shape: (1, 5, num_anchors)

        # YOLOv8 output layout: [batch, (cx,cy,w,h,conf*cls...), anchors]
        out = raw[0]                            # (5, num_anchors)
        if out.shape[0] == 5:                   # (5, N) → transpose to (N, 5)
            out = out.T

        boxes, scores = [], []
        sx, sy = w / sz, h / sz
        for row in out:
            conf = float(row[4])
            if conf < self._yolo_conf_threshold:
                continue
            cx, cy, bw, bh = row[0] * sx, row[1] * sy, row[2] * sx, row[3] * sy
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            boxes.append([x1, y1, int(bw), int(bh)])
            scores.append(conf)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self._yolo_conf_threshold, self._yolo_nms_threshold)
        detections = []
        for i in (indices.flatten() if len(indices) else []):
            x, y, bw, bh = boxes[i]
            detections.append({
                'bbox': (x, y, bw, bh),
                'confidence': scores[i],
                'centroid': (x + bw // 2, y + bh // 2),
                'type': 'yolo',
            })
        return detections

    def _detect_yolo11n_coco(self, frame):
        """YOLO11n COCO inference via cv2.dnn — filters for bird class (COCO class 14).

        COCO YOLO11n ONNX output layout: (1, 84, num_anchors)
          rows 0-3  : cx, cy, w, h (normalised to input size)
          rows 4-83 : per-class scores for all 80 COCO classes
          bird = class index 14 → output row 4+14 = 18
        """
        BIRD_CLASS_IDX = 14
        h, w = frame.shape[:2]
        sz = self._yolo_input_size
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (sz, sz), swapRB=True, crop=False)
        self._yolo11n_net.setInput(blob)
        raw = self._yolo11n_net.forward()   # (1, 84, num_anchors)

        out = raw[0]                         # (84, num_anchors)
        if out.shape[0] == 84:               # (84, N) → transpose to (N, 84)
            out = out.T

        boxes, scores = [], []
        sx, sy = w / sz, h / sz
        for row in out:
            conf = float(row[4 + BIRD_CLASS_IDX])
            if conf < self._yolo_conf_threshold:
                continue
            cx, cy, bw, bh = row[0] * sx, row[1] * sy, row[2] * sx, row[3] * sy
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            boxes.append([x1, y1, int(bw), int(bh)])
            scores.append(conf)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self._yolo_conf_threshold, self._yolo_nms_threshold)
        detections = []
        for i in (indices.flatten() if len(indices) else []):
            x, y, bw, bh = boxes[i]
            detections.append({
                'bbox': (x, y, bw, bh),
                'confidence': scores[i],
                'centroid': (x + bw // 2, y + bh // 2),
                'type': 'yolo11n_coco',
            })
        return detections

    def _detect_motion_birds(self, frame, mask):
        """Shape-aware motion detection with 5-feature confidence scoring.

        Grayscale and frame mean are computed once per frame.
        Hull, hull_area, and perimeter are computed ONCE per contour and passed
        to both the hard shape gate and the confidence scorer — no duplicate work.
        Fragment NMS merges blobs from the same bird before returning.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_mean = float(np.mean(gray))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_contour_area < area < self.max_contour_area):
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Pre-compute shape descriptors once — shared by gate and scorer
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            perimeter = cv2.arcLength(contour, True)

            if not self._is_valid_bird_shape(area, w, h, hull_area, perimeter):
                continue

            gray_roi = gray[y:y + h, x:x + w]
            features = self._score_shape_features(area, w, h, gray_roi, frame_mean, hull_area, perimeter)
            confidence = self._calculate_confidence_v2(features)

            if confidence > self.confidence_threshold:
                raw.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'centroid': (x + w // 2, y + h // 2),
                    'type': 'motion',
                })

        # Merge fragments from the same bird before returning
        return self._merge_detections_nms(raw)

    def _detect_contour_motion(self, frame, mask):
        """Motion-only detection optimized for tiny, fast-moving swiftlets.

        No shape validation at all — any moving blob within the configured
        area range is treated as a bird candidate.  The MOG2 mask is combined
        with a raw frame-difference mask for extra sensitivity, then morphology
        cleans up salt-and-pepper noise before contour extraction.

        Config keys:
          cm_min_area       — minimum blob area (default 12 px²)
          cm_max_area       — maximum blob area (default 700 px²)
          cm_diff_threshold — pixel-level diff threshold for frame diff (default 12)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Combine MOG2 mask with frame difference for extra motion emphasis
        combined = mask.copy()
        if self._prev_gray is not None:
            diff = cv2.absdiff(self._prev_gray, gray)
            _, diff_mask = cv2.threshold(diff, self.cm_diff_threshold, 255, cv2.THRESH_BINARY)
            combined = cv2.bitwise_or(combined, diff_mask)
        self._prev_gray = gray

        # Morphological cleanup: open removes isolated noise pixels, dilate bridges
        # nearby blobs that belong to the same tiny bird
        motion_mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.morph_kernel)
        motion_mask = cv2.dilate(motion_mask, self.morph_kernel, iterations=2)

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.cm_min_area < area < self.cm_max_area):
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Confidence: peak in the middle of the expected area range, fades toward edges
            mid_area = (self.cm_min_area + self.cm_max_area) / 2.0
            span     = (self.cm_max_area - self.cm_min_area) / 2.0
            conf = max(0.55, 1.0 - 0.45 * abs(area - mid_area) / max(1.0, span))

            detections.append({
                'bbox':       (x, y, w, h),
                'confidence': conf,
                'centroid':   (x + w // 2, y + h // 2),
                'type':       'contour_motion',
            })

        return self._merge_detections_nms(detections)

    def _score_shape_features(self, area, w, h, gray_roi, frame_mean, hull_area, perimeter):
        """Compute 5 normalized [0,1] shape feature scores for a contour.

        Accepts pre-computed hull_area and perimeter — no duplicate work.
        Darkness uses a clean relative ratio (frame_mean - roi_mean) / frame_mean
        which is stable in any room brightness, unlike the old saturating formula.
        """
        # 1. Area score — calibrated from 3,766 real swiftlet annotations
        #    IQR (640×480 scaled): ~1000–3000 px²;  p95: ~8000 px²
        if 1000 <= area <= 3000:
            area_score = 1.0
        elif area < 1000:
            area_score = 0.7 + 0.3 * (area - self.min_contour_area) / max(1, 1000 - self.min_contour_area)
        else:
            area_score = max(0.4, 1.0 - 0.6 * (area - 3000) / max(1, self.max_contour_area - 3000))

        # 2. Aspect ratio score — calibrated from real data
        #    Median AR = 0.85 (nearly square), IQR = 0.72–1.08, p95 = 1.70
        #    Birds are NOT wide-winged crescents at this camera angle/distance
        ar = w / h if h > 0 else 1.0
        if 0.7 <= ar <= 1.1:
            ar_score = 1.0
        elif 0.5 <= ar < 0.7 or 1.1 < ar <= 1.7:
            ar_score = 0.7
        elif 1.7 < ar <= self.shape_aspect_max:
            ar_score = 0.5
        else:
            ar_score = 0.3

        # 3. Solidity score — convex hull fill ratio (pre-computed)
        solidity = (area / hull_area) if hull_area > 0 else 0.0
        sol_score = min(1.0, solidity / 0.65)

        # 4. Compactness score — 4π·area/perimeter² (pre-computed)
        compactness = (4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        comp_score = min(1.0, compactness / 0.30)

        # 5. Darkness score — relative ratio: stable across all room brightness levels
        if gray_roi.size > 0:
            roi_mean = float(np.mean(gray_roi))
            darkness_score = max(0.0, min(1.0, (frame_mean - roi_mean) / max(1.0, frame_mean)))
        else:
            darkness_score = 0.5

        return {
            'area': area_score,
            'ar': ar_score,
            'solidity': sol_score,
            'compactness': comp_score,
            'darkness': darkness_score,
        }

    def _calculate_confidence_v2(self, features):
        """Weighted 5-feature confidence score tuned for swiftlets."""
        return max(0.05, min(1.0, (
            features['area']        * self.conf_w_area +
            features['ar']          * self.conf_w_aspect +
            features['solidity']    * self.conf_w_solidity +
            features['compactness'] * self.conf_w_compactness +
            features['darkness']    * self.conf_w_darkness
        )))

    def _merge_detections_nms(self, detections):
        """IoU + centroid-distance NMS to merge BGS fragments from the same bird.

        Sorted by confidence desc so the strongest detection absorbs its neighbors.
        Two detections are merged when IoU > nms_iou_threshold OR centroid distance
        < nms_merge_distance (handles fragments that don't overlap at all).
        """
        if len(detections) <= 1:
            return detections

        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        suppressed = [False] * len(detections)
        result = []

        for i, di in enumerate(detections):
            if suppressed[i]:
                continue
            xi, yi, wi, hi = di['bbox']
            cx_i, cy_i = di['centroid']
            merged_bbox = [xi, yi, xi + wi, yi + hi]  # x1,y1,x2,y2

            for j in range(i + 1, len(detections)):
                if suppressed[j]:
                    continue
                dj = detections[j]
                xj, yj, wj, hj = dj['bbox']
                cx_j, cy_j = dj['centroid']

                # Centroid distance check (cheap — do first)
                dist = math.hypot(cx_i - cx_j, cy_i - cy_j)
                if dist < self.nms_merge_distance:
                    suppressed[j] = True
                else:
                    # IoU check
                    ix1 = max(xi, xj); iy1 = max(yi, yj)
                    ix2 = min(xi + wi, xj + wj); iy2 = min(yi + hi, yj + hj)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter > 0:
                        union = wi * hi + wj * hj - inter
                        if union > 0 and (inter / union) > self.nms_iou_threshold:
                            suppressed[j] = True

                if suppressed[j]:
                    # Expand merged bbox to union of both
                    merged_bbox[0] = min(merged_bbox[0], xj)
                    merged_bbox[1] = min(merged_bbox[1], yj)
                    merged_bbox[2] = max(merged_bbox[2], xj + wj)
                    merged_bbox[3] = max(merged_bbox[3], yj + hj)

            mx, my = merged_bbox[0], merged_bbox[1]
            mw = merged_bbox[2] - merged_bbox[0]
            mh = merged_bbox[3] - merged_bbox[1]
            result.append({
                'bbox': (mx, my, mw, mh),
                'confidence': di['confidence'],
                'centroid': (mx + mw // 2, my + mh // 2),
                'type': di['type'],
            })

        return result
    
    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default parameters.")
            return {}
    
    def _is_valid_bird_shape(self, area, width, height, hull_area, perimeter):
        """Hard-gate shape validation — rejects non-bird contours before confidence scoring.

        Accepts pre-computed hull_area and perimeter so convex hull is never
        computed twice for the same contour.
        """
        if width < 3 or height < 3 or width > 300 or height > 300:
            return False

        # Aspect ratio: swiftlets in flight are wider than tall (crescent wings)
        ar = width / height if height > 0 else 0
        if ar < self.shape_aspect_min or ar > self.shape_aspect_max:
            return False

        # Solidity: bird bodies are convex-ish; shadows have deep concavities
        if hull_area > 0 and (area / hull_area) < self.shape_min_solidity:
            return False

        # Compactness (circularity): eliminates thin wires, ledge edges
        if perimeter > 0:
            compactness = 4.0 * math.pi * area / (perimeter * perimeter)
            if compactness < self.shape_min_compactness:
                return False

        # Extent: contour area vs bounding rect — rejects scraggly noise clusters
        if (area / (width * height)) < self.shape_min_extent:
            return False

        return True
    
    def _get_center_box(self, frame_width, frame_height):
        """Create center box based on configurable ratios."""
        box_w = int(frame_width * self.center_box_width_ratio)
        box_h = int(frame_height * self.center_box_height_ratio)
        x1 = max(0, (frame_width - box_w) // 2)
        y1 = max(0, (frame_height - box_h) // 2)
        x2 = min(frame_width - 1, x1 + box_w)
        y2 = min(frame_height - 1, y1 + box_h)
        return (x1, y1, x2, y2)

    def _is_inside_center_box(self, point):
        """Check whether point lies in the center box."""
        x, y = point
        x1, y1, x2, y2 = self.center_box
        return x1 <= x <= x2 and y1 <= y <= y2

    def _get_expanded_center_box(self, frame_width, frame_height):
        """Expand center box with margin to better capture fast crossing."""
        x1, y1, x2, y2 = self.center_box
        margin = int(min(frame_width, frame_height) * self.center_gate_margin_ratio)
        ex1 = max(0, x1 - margin)
        ey1 = max(0, y1 - margin)
        ex2 = min(frame_width - 1, x2 + margin)
        ey2 = min(frame_height - 1, y2 + margin)
        return (ex1, ey1, ex2, ey2)

    def _line_intersects_box(self, p1, p2, box):
        """Check if movement segment intersects box."""
        if p1 is None or p2 is None:
            return False
        x1, y1, x2, y2 = box
        rect = (x1, y1, max(1, x2 - x1 + 1), max(1, y2 - y1 + 1))
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]), int(p2[1]))
        intersects, _, _ = cv2.clipLine(rect, pt1, pt2)
        return intersects

    def _is_count_allowed(self, tracker, now_time, movement_px):
        """Debounce count events to prevent spikes."""
        last_update_source = tracker.get('last_update_source', 'detection')
        lost_frames = tracker.get('lost_frames', 0)
        if lost_frames > self.count_max_lost_frames:
            return False
        if movement_px < self.counting_min_displacement_px:
            return False

        confidence = tracker.get('confidence', 0.0)
        if last_update_source == 'detection':
            if confidence < self.counting_confidence_threshold:
                return False
        else:
            if not self.count_allow_prediction:
                return False
            last_det_frame = tracker.get('last_detection_frame', -10_000)
            if (self.frame_count - last_det_frame) > self.count_recent_detection_window:
                return False
            vx, vy = tracker.get('velocity', (0.0, 0.0))
            speed = float(np.sqrt(vx * vx + vy * vy))
            if speed < self.count_prediction_min_speed:
                return False
            if confidence < self.count_relaxed_confidence_threshold:
                return False

        last_counted_time = tracker.get('last_counted_time')
        if last_counted_time is not None:
            if (now_time - last_counted_time) < self.tracker_count_cooldown_seconds:
                return False

        if (now_time - self._last_count_time) < self.global_count_min_interval_seconds:
            return False

        return True

    def _update_crossing_count(self, tracker, bbox):
        """Update counter based on robust crossing checks (anti-spike)."""
        x, y, w, h = bbox
        center_now = (x + w // 2, y + h // 2)
        center_prev = tracker.get('prev_center')
        inside_now = self._is_inside_center_box(center_now)
        inside_before = tracker.get('inside_center_box')

        if inside_before is None or center_prev is None:
            tracker['inside_center_box'] = inside_now
            tracker['prev_center'] = center_now
            return

        frame_w = max(1, self.width)
        frame_h = max(1, self.height)
        expanded_box = self._get_expanded_center_box(frame_w, frame_h)
        crossed_gate = self._line_intersects_box(center_prev, center_now, expanded_box)

        movement_px = float(np.sqrt(
            (center_now[0] - center_prev[0]) ** 2 +
            (center_now[1] - center_prev[1]) ** 2
        ))
        now_time = time.time()

        did_count = False
        if inside_before != inside_now:
            if self._is_count_allowed(tracker, now_time, movement_px):
                if inside_before and not inside_now:
                    self.crossing_count += 1
                    self.cross_in_to_out += 1
                    did_count = True
                elif (not inside_before) and inside_now:
                    self.crossing_count = max(0, self.crossing_count - 1)
                    self.cross_out_to_in += 1
                    did_count = True
        elif crossed_gate and (not inside_before) and (not inside_now):
            if self._is_count_allowed(tracker, now_time, movement_px):
                gate_cx = (self.center_box[0] + self.center_box[2]) / 2.0
                gate_cy = (self.center_box[1] + self.center_box[3]) / 2.0
                prev_dist = np.sqrt((center_prev[0] - gate_cx) ** 2 + (center_prev[1] - gate_cy) ** 2)
                curr_dist = np.sqrt((center_now[0] - gate_cx) ** 2 + (center_now[1] - gate_cy) ** 2)
                if curr_dist > prev_dist:
                    self.crossing_count += 1
                    self.cross_in_to_out += 1
                    did_count = True
                elif curr_dist < prev_dist:
                    self.crossing_count = max(0, self.crossing_count - 1)
                    self.cross_out_to_in += 1
                    did_count = True

        if did_count:
            tracker['last_counted_time'] = now_time
            self._last_count_time = now_time

        tracker['inside_center_box'] = inside_now
        tracker['prev_center'] = center_now

    def _smooth_bbox(self, prev_bbox, new_bbox):
        """Apply EMA smoothing to bbox to reduce jitter."""
        px, py, pw, ph = prev_bbox
        nx, ny, nw, nh = new_bbox
        alpha = self.bbox_smooth_alpha

        sx = int(alpha * px + (1 - alpha) * nx)
        sy = int(alpha * py + (1 - alpha) * ny)
        sw = int(alpha * pw + (1 - alpha) * nw)
        sh = int(alpha * ph + (1 - alpha) * nh)

        sw = max(1, sw)
        sh = max(1, sh)
        return (sx, sy, sw, sh)

    def _is_reasonable_bbox_transition(self, prev_bbox, new_bbox):
        """Reject abrupt bbox jumps/scale changes that usually cause wild boxes."""
        px, py, pw, ph = prev_bbox
        nx, ny, nw, nh = new_bbox

        prev_center = (px + pw / 2.0, py + ph / 2.0)
        new_center = (nx + nw / 2.0, ny + nh / 2.0)
        center_distance = np.sqrt(
            (prev_center[0] - new_center[0]) ** 2 +
            (prev_center[1] - new_center[1]) ** 2
        )
        if center_distance > (self.tracker_max_distance * 1.25):
            return False

        width_ratio = nw / max(1.0, float(pw))
        height_ratio = nh / max(1.0, float(ph))
        max_change = self.max_bbox_size_change
        min_change = 1.0 / max_change
        if not (min_change <= width_ratio <= max_change):
            return False
        if not (min_change <= height_ratio <= max_change):
            return False

        return True

    def _create_kalman_filter(self, cx, cy):
        """Kalman filter with constant-velocity model for swiftlet tracking.

        State:  [cx, cy, vx, vy]   (centroid + velocity)
        Measurement: [cx, cy]

        Replaces MOSSE entirely. MOSSE was initialized per-tracker (~3ms each)
        but tracker.update() was never called (use_tracker_prediction=False),
        making it pure wasted CPU. Kalman predict+correct costs ~0.03ms total.

        Higher velocity process noise (kalman_pn_vel) lets the filter adapt
        quickly to swiftlets' abrupt direction changes.
        """
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        kf.processNoiseCov = np.diag([
            self.kalman_pn_pos, self.kalman_pn_pos,
            self.kalman_pn_vel, self.kalman_pn_vel,
        ]).astype(np.float32)
        kf.measurementNoiseCov = np.diag([
            self.kalman_mn, self.kalman_mn,
        ]).astype(np.float32)
        kf.errorCovPost = np.diag([10., 10., 100., 100.]).astype(np.float32)
        kf.statePost = np.array([[float(cx)], [float(cy)], [0.], [0.]], dtype=np.float32)
        return kf

    def _create_tracker(self):
        """Legacy stub — kept for API compatibility but returns None.
        Tracking is now done via Kalman filter (_create_kalman_filter).
        """
        return None
    
    def _update_pending_pool(self, unmatched_detections):
        """Pre-tracking temporal vote gate.

        A detection must appear in >= pending_vote_frames frames within a
        pending_vote_window frame window before a tracker is created.
        This eliminates single-frame specular glints, insects, and sensor noise
        that would otherwise create trackers lasting 15 frames near the count line.

        Returns the list of promoted detections ready for tracker creation.
        """
        for det in unmatched_detections:
            cx, cy = det['centroid']
            matched = False
            for p in self._pending_pool:
                pcx, pcy = p['centroid']
                if math.hypot(cx - pcx, cy - pcy) < self.pending_max_distance:
                    p['votes'] += 1
                    p['last_frame'] = self.frame_count
                    p['centroid'] = (cx, cy)   # track to latest position
                    p['bbox'] = det['bbox']
                    p['confidence'] = max(p['confidence'], det['confidence'])
                    matched = True
                    break
            if not matched:
                self._pending_pool.append({
                    'centroid': (cx, cy),
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'votes': 1,
                    'last_frame': self.frame_count,
                })

        # Expire stale entries (not seen in pending_vote_window frames)
        self._pending_pool = [
            p for p in self._pending_pool
            if (self.frame_count - p['last_frame']) <= self.pending_vote_window
        ]

        # Promote entries that have enough votes
        promoted = [p for p in self._pending_pool if p['votes'] >= self.pending_vote_frames]
        # Remove promoted entries from pool
        self._pending_pool = [p for p in self._pending_pool if p['votes'] < self.pending_vote_frames]
        return promoted

    def update_trackers(self, frame, detections):
        """Detection-based tracker management with Kalman prediction + temporal vote gate."""
        self.center_box = self._get_center_box(frame.shape[1], frame.shape[0])
        updated_trackers = []
        used_detections = []

        # ── Update existing trackers ──────────────────────────────────────────
        for tracker in self.trackers:
            prev_x, prev_y, prev_w, prev_h = tracker['bbox']
            prev_center = (prev_x + prev_w // 2, prev_y + prev_h // 2)
            lost_frames = tracker.get('lost_frames', 0)
            velocity = tracker.get('velocity', (0.0, 0.0))
            kf = tracker.get('kalman')

            # Use Kalman predicted center as match target (more accurate than
            # raw velocity damping, especially after 1-2 lost frames)
            if kf is not None and lost_frames > 0:
                predicted = kf.predict()
                expected_center = (int(predicted[0]), int(predicted[1]))
                # Undo internal state advance — we'll re-predict only if no match
                kf.statePost = kf.statePre.copy()
            else:
                expected_center = (
                    int(prev_center[0] + velocity[0]),
                    int(prev_center[1] + velocity[1])
                ) if lost_frames > 0 else prev_center

            allowed_distance = min(
                self.tracker_max_distance_cap,
                self.tracker_max_distance * (1.0 + lost_frames * self.tracker_distance_growth)
            )

            best_match = None
            best_distance = float('inf')
            for i, detection in enumerate(detections):
                if i in used_detections:
                    continue
                det_center = detection['centroid']
                dist = math.hypot(expected_center[0] - det_center[0],
                                  expected_center[1] - det_center[1])
                if dist < best_distance and dist < allowed_distance:
                    best_distance = dist
                    best_match = i

            if best_match is not None:
                detection = detections[best_match]
                used_detections.append(best_match)
                candidate_bbox = detection['bbox']

                if not self._is_reasonable_bbox_transition(tracker['bbox'], candidate_bbox):
                    tracker['lost_frames'] = lost_frames + 1
                    tracker['confidence'] *= 0.95
                    if (tracker['confidence'] > 0.3 and
                            tracker['lost_frames'] <= self.tracker_max_lost_frames):
                        updated_trackers.append(tracker)
                    continue

                tracker['bbox'] = self._smooth_bbox(tracker['bbox'], candidate_bbox)
                tracker['confidence'] = detection['confidence']
                tracker['detection_type'] = detection.get('type', tracker.get('detection_type', 'motion'))

                new_center = (
                    tracker['bbox'][0] + tracker['bbox'][2] // 2,
                    tracker['bbox'][1] + tracker['bbox'][3] // 2,
                )

                # Kalman correct step
                if kf is not None:
                    meas = np.array([[float(new_center[0])], [float(new_center[1])]], dtype=np.float32)
                    kf.correct(meas)
                    state = kf.statePost
                    tracker['velocity'] = (float(state[2]), float(state[3]))
                else:
                    tracker['velocity'] = (
                        float(new_center[0] - prev_center[0]),
                        float(new_center[1] - prev_center[1]),
                    )

                tracker['last_update_source'] = 'detection'
                tracker['last_detection_frame'] = self.frame_count
                tracker['lost_frames'] = 0
                self._update_crossing_count(tracker, tracker['bbox'])
                updated_trackers.append(tracker)
                continue

            # ── No detection matched: predict position ────────────────────────
            tracker['lost_frames'] = lost_frames + 1

            if kf is not None:
                # Kalman predict advances the state by F matrix
                predicted = kf.predict()
                pred_cx = float(predicted[0])
                pred_cy = float(predicted[1])
                pred_vx = float(predicted[2])
                pred_vy = float(predicted[3])
                px = max(0, min(self.width - prev_w, int(pred_cx - prev_w / 2)))
                py = max(0, min(self.height - prev_h, int(pred_cy - prev_h / 2)))
                tracker['bbox'] = (px, py, prev_w, prev_h)
                tracker['velocity'] = (pred_vx, pred_vy)
            else:
                vx, vy = velocity
                damping = float(self.tracker_velocity_damping)
                px = max(0, min(self.width - prev_w, int(prev_x + vx)))
                py = max(0, min(self.height - prev_h, int(prev_y + vy)))
                tracker['bbox'] = (px, py, prev_w, prev_h)
                tracker['velocity'] = (vx * damping, vy * damping)

            tracker['confidence'] *= 0.96
            tracker['last_update_source'] = 'prediction'
            self._update_crossing_count(tracker, tracker['bbox'])

            if (tracker['confidence'] > self.tracker_min_confidence_keep and
                    tracker['lost_frames'] <= self.tracker_max_lost_frames):
                updated_trackers.append(tracker)

        # ── Create new trackers via temporal vote gate ────────────────────────
        unmatched = [
            detections[i] for i in range(len(detections))
            if i not in used_detections and detections[i]['confidence'] > 0.5
        ]
        promoted = self._update_pending_pool(unmatched)

        new_trackers_created = 0
        for det in promoted:
            x, y, w, h = det['bbox']
            if not (w > 0 and h > 0 and x >= 0 and y >= 0
                    and x + w <= self.width and y + h <= self.height):
                continue
            if len(updated_trackers) >= self.max_birds_per_frame:
                break

            cx, cy = x + w // 2, y + h // 2
            self.bird_id_counter += 1
            new_trackers_created += 1
            updated_trackers.append({
                'tracker': None,
                'kalman': self._create_kalman_filter(cx, cy) if self.use_kalman_filter else None,
                'id': self.bird_id_counter,
                'bbox': (x, y, w, h),
                'confidence': det['confidence'],
                'detection_type': det.get('type', 'motion'),
                'lost_frames': 0,
                'inside_center_box': self._is_inside_center_box((cx, cy)),
                'prev_center': (cx, cy),
                'velocity': (0.0, 0.0),
                'last_counted_time': None,
                'last_update_source': 'detection',
                'last_detection_frame': self.frame_count,
            })

        if new_trackers_created > 0:
            print(f"Frame {self.frame_count}: Created {new_trackers_created} new trackers")

        self.trackers = updated_trackers
        self.bird_count = len(self.trackers)
    
    def draw_annotations(self, frame):
        """Draw bounding boxes and labels on the frame"""
        frame_height, frame_width = frame.shape[:2]
        self.center_box = self._get_center_box(frame_width, frame_height)

        # Tick FPS counter
        now = time.time()
        self._fps_times.append(now)
        if len(self._fps_times) >= 2:
            fps_val = (len(self._fps_times) - 1) / (self._fps_times[-1] - self._fps_times[0])
        else:
            fps_val = 0.0
        x1, y1, x2, y2 = self.center_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        for tracker in self.trackers:
            x, y, w, h = tracker['bbox']
            confidence = tracker['confidence']
            bird_id = tracker['id']

            if confidence < self.display_confidence_threshold:
                continue
            
            # Draw bounding box with different colors for motion vs static
            detection_type = tracker.get('detection_type', 'motion')
            if detection_type == 'static':
                color = (255, 0, 255)  # Magenta for static detection
            elif confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence motion
            else:
                color = (0, 255, 255)  # Yellow for medium confidence motion
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with detection type
            label = f"Swiftlet: {bird_id%100} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            
            cv2.rectangle(frame, (x, label_y - label_size[1] - 5), 
                         (x + label_size[0] + 5, label_y + 5), color, -1)
            cv2.putText(frame, label, (x + 2, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        base_width = 1920
        text_scale = 1.2 * (frame_width / base_width)
        text_scale_number = 1.8 * (frame_width / base_width)
        counter_text_thickness = int(self.config.get('counter_text_thickness', 2))
        counter_number_thickness = int(self.config.get('counter_number_thickness', 3))
        margin = 20

        # --- Top Right: FPS ---
        fps_label = f"FPS: {fps_val:.1f}"
        ts_fps, _ = cv2.getTextSize(fps_label, font, text_scale, counter_text_thickness)
        x_fps = frame_width - margin - ts_fps[0]
        y_fps = margin + ts_fps[1]
        fps_color = (0, 255, 0) if fps_val >= 8 else (0, 165, 255) if fps_val >= 4 else (0, 0, 255)
        cv2.putText(frame, fps_label, (x_fps, y_fps),
            font, text_scale, fps_color, counter_text_thickness)

        # --- Bottom Left: Device Name + Counter Burung Walet ---
        count_prefix = "Counter Burung Walet: "
        count_number = str(self.crossing_count)

        text_size_prefix, _ = cv2.getTextSize(count_prefix, font, text_scale, 1)
        text_size_number, _ = cv2.getTextSize(count_number, font, text_scale_number, 1)
        text_size_title, _ = cv2.getTextSize(self.device_name, font, text_scale, 1)

        line_h = int(text_size_prefix[1] * 1.6)

        # Stack from bottom: count row, then title above it
        y_bl_count = frame_height - margin
        y_bl_title = y_bl_count - line_h

        x_bl = margin

        cv2.putText(frame, self.device_name, (x_bl, y_bl_title),
            font, text_scale, (255, 255, 255), counter_text_thickness)
        cv2.putText(frame, count_prefix, (x_bl, y_bl_count),
            font, text_scale, (255, 255, 255), counter_text_thickness)
        cv2.putText(frame, count_number, (x_bl + text_size_prefix[0], y_bl_count),
            font, text_scale_number, (50, 255, 50), counter_number_thickness)

        # --- Bottom Right: Live Counter (Dalam & Luar) ---
        visible_trackers = [t for t in self.trackers if t.get('confidence', 0) >= self.display_confidence_threshold]
        live_inside = sum(1 for t in visible_trackers if t.get('inside_center_box', False))
        live_outside = len(visible_trackers) - live_inside

        live_total = len(visible_trackers)
        live_title  = f"Live Counter: {live_total}"
        live_dalam  = f"Dalam: {live_inside}"
        live_luar   = f"Luar:  {live_outside}"

        # Measure widths for right-alignment
        ts_live_title, _ = cv2.getTextSize(live_title, font, text_scale, 1)
        ts_live_dalam, _ = cv2.getTextSize(live_dalam, font, text_scale, 1)
        ts_live_luar,  _ = cv2.getTextSize(live_luar,  font, text_scale, 1)

        max_w = max(ts_live_title[0], ts_live_dalam[0], ts_live_luar[0])
        x_br = frame_width - margin - max_w

        # Stack from bottom: luar, dalam, title
        y_br_luar  = frame_height - margin
        y_br_dalam = y_br_luar  - line_h
        y_br_title = y_br_dalam - line_h

        cv2.putText(frame, live_title, (x_br, y_br_title),
            font, text_scale, (255, 255, 0), counter_text_thickness)
        cv2.putText(frame, live_dalam, (x_br, y_br_dalam),
            font, text_scale, (255, 255, 255), counter_text_thickness)
        cv2.putText(frame, live_luar,  (x_br, y_br_luar),
            font, text_scale, (255, 255, 255), counter_text_thickness)

        return frame
    
    
    def process_video(self, show_preview=True):
        """Enhanced video processing with temporal consistency - only for video file mode"""
        if self.streaming_mode:
            print("Warning: process_video() should not be called in streaming mode")
            return
            
        if not self.cap or not self.out:
            print("Error: Video capture or writer not initialized")
            return
            
        self.frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Apply background subtraction
            mask = self.bg_subtractor.apply(frame)
            
            # Remove noise - simple like original
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Detect birds
            detections = self.detect_birds(frame, mask)
            
            # Update trackers
            self.update_trackers(frame, detections)
            
            # Update statistics
            self.detection_history.append(len(detections))
            
            # Draw annotations
            annotated_frame = frame.copy()
            annotated_frame = self.draw_annotations(annotated_frame)
            
            # Write frame to output video
            self.out.write(annotated_frame)
            
            # Show preview if requested
            if show_preview:
                preview_scale = self.config.get('preview_scale', 0.6)
                preview_w = int(self.width * preview_scale)
                preview_h = int(self.height * preview_scale)
                preview = cv2.resize(annotated_frame, (preview_w, preview_h))
                cv2.imshow('Swiftlet Counter', preview)
                
                # Show enhanced mask for debugging
                mask_preview = cv2.resize(mask, (400, 300))
                cv2.imshow('Motion Mask', mask_preview)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress with enhanced statistics
            if self.frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = self.frame_count / elapsed
                avg_detections = np.mean(self.detection_history) if self.detection_history else 0
                print(f"Processed {self.frame_count} frames at {fps:.2f} FPS. "
                      f"Current bird count: {self.bird_count}, "
                      f"Avg detections: {avg_detections:.1f}")
        
        # Cleanup
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Output saved to: {self.output_path}")
        print(f"Maximum birds detected: {self.bird_id_counter}")
        print(f"Average detections per frame: {np.mean(self.detection_history):.2f}")
        
        # Save statistics if requested
        if self.config.get('save_statistics', False):
            self._save_statistics()
    def set_device_name(self, name):
        """Set the device name for display and statistics"""
        self.device_name = name
        print(f"Device name set to: {self.device_name}")
    
    def _preprocess_mask(self, mask):
        """Swiftlet-optimized mask preprocessing.

        MOG2 marks shadow pixels as 127 and foreground as 255.
        Swiftlets are dark birds — their silhouette is real foreground (255).
        Shadows are lighter gradients; killing them (127→0) removes ~60% of
        false contours without losing bird detections.

        The CLOSE kernel merges the typical 2-4 fragment blobs that BGS
        produces from a single bird's body and wings into one contour.
        """
        # Kill MOG2 shadow pixels — swiftlets are 255 foreground, not 127 shadow
        mask[mask == 127] = 0

        # Only confident foreground passes
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        # OPEN: remove pepper noise (stray 1-3px hits)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)

        # CLOSE: merge nearby fragments from same bird into one blob
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        # Light blur then re-binarize to smooth blob edges
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask
    
    def _save_statistics(self):
        """Save processing statistics to file"""
        stats = {
            'total_frames': self.frame_count,
            'max_birds_detected': self.bird_id_counter,
            'detection_history': list(self.detection_history),
            'config_used': self.config
        }
        
        stats_path = self.output_path.replace('.mp4', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to: {stats_path}")