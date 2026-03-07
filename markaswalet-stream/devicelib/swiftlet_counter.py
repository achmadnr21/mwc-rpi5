import cv2
import numpy as np
import time
import json

from ultralytics import YOLO
import supervision as sv


class YOLOSwiftletCounter:
    def __init__(self, config_path="config.json", streaming_mode=True, device_name="RBW Lantai 1"):
        self.streaming_mode = streaming_mode
        self.device_name = device_name

        # Load configuration
        self.config = self._load_config(config_path)

        # Load YOLO11n OpenVINO model
        model_path = self.config.get("model_path", "models/yolo11n_openvino_model")
        print(f"[Counter] Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = self.config.get("confidence_threshold", 0.35)
        self.bird_classes = self.config.get("bird_classes", [14])  # COCO class 14 = bird

        # Init ByteTrack
        self.tracker = sv.ByteTrack()

        # Init LineZone (horizontal midline by default; adjust in config for your camera angle)
        line_start_cfg = self.config.get("line_start", [50, 240])
        line_end_cfg = self.config.get("line_end", [590, 240])
        line_start = sv.Point(line_start_cfg[0], line_start_cfg[1])
        line_end = sv.Point(line_end_cfg[0], line_end_cfg[1])
        self.line_zone = sv.LineZone(start=line_start, end=line_end)

        # Annotators
        self.box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)
        self.line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

        # Count and timing state
        self.total_count = 0
        self.frame_count = 0
        self.last_api_report_time = time.time()
        self.api_report_interval = self.config.get("api_report_interval", 30)

        # Lazy-init Device reference for API reporting
        self._device = None

        print(f"[Counter] YOLOSwiftletCounter ready. Line: {line_start_cfg} → {line_end_cfg}")

    def _load_config(self, config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[Counter] Config '{config_path}' not found. Using defaults.")
            return {}

    def _get_device(self):
        if self._device is None:
            try:
                from devicelib.device import Device
                self._device = Device()
            except Exception as e:
                print(f"[Counter] Warning: Could not init Device for API reporting: {e}")
        return self._device

    def process_frame(self, frame) -> np.ndarray:
        self.frame_count += 1

        # 1. YOLO inference — filter to bird class only
        results = self.model(
            frame,
            conf=self.conf_threshold,
            classes=self.bird_classes,
            verbose=False,
        )[0]

        # 2. Convert to supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # 3. ByteTrack update
        detections = self.tracker.update_with_detections(detections)

        # 4. LineZone crossing check
        self.line_zone.trigger(detections)

        # 5. Total line crossings (in + out = all birds that crossed)
        self.total_count = self.line_zone.in_count + self.line_zone.out_count

        # 6. Annotate frame
        annotated = self._draw_annotations(frame, detections)

        # 7. Periodic API report
        now = time.time()
        if now - self.last_api_report_time >= self.api_report_interval:
            self._report_count_to_api()
            self.last_api_report_time = now

        return annotated

    def _draw_annotations(self, frame, detections):
        annotated = frame.copy()

        # Bounding boxes
        annotated = self.box_annotator.annotate(scene=annotated, detections=detections)

        # Track ID labels
        if len(detections) > 0 and detections.tracker_id is not None:
            labels = [f"#{tid}" for tid in detections.tracker_id]
            annotated = self.label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )

        # Counting line
        annotated = self.line_annotator.annotate(annotated, line_counter=self.line_zone)

        # Count overlay (matches original style)
        frame_h, frame_w = annotated.shape[:2]
        font = cv2.FONT_HERSHEY_DUPLEX
        base_width = 1920
        text_scale = 1.2 * (frame_w / base_width)
        x_pos = int(frame_w * 0.65)
        y_title = int(frame_h * 0.06)
        y_count = int(frame_h * 0.11)

        cv2.putText(annotated, self.device_name, (x_pos, y_title), font, text_scale, (255, 255, 255), 2)

        count_prefix = "Total Burung Walet: "
        cv2.putText(annotated, count_prefix, (x_pos, y_count), font, text_scale, (255, 255, 255), 2)

        prefix_size, _ = cv2.getTextSize(count_prefix, font, text_scale, 2)
        number_x = x_pos + prefix_size[0]
        text_scale_number = 1.8 * (frame_w / base_width)
        cv2.putText(
            annotated, str(self.total_count),
            (number_x, y_count), font, text_scale_number, (50, 255, 50), 2
        )

        return annotated

    def _report_count_to_api(self):
        device = self._get_device()
        if device is None:
            return
        try:
            success = device.report_count(self.total_count)
            status = "OK" if success else "FAILED"
            print(f"[Counter] API report count={self.total_count} → {status}")
        except Exception as e:
            print(f"[Counter] API report error: {e}")
