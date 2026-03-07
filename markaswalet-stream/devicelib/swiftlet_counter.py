import cv2
import numpy as np
import time
import json
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
        center_box_height_ratio=None
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
        
        # Simple background subtractor
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
        
        # Motion history for temporal consistency
        self.motion_history_frames = self.config.get('motion_history_frames', 5)
        self.frame_buffer = deque(maxlen=self.motion_history_frames)
        
        # Morphological kernel
        kernel_size = self.config.get('morphology_kernel_size', 3)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Statistics
        self.frame_count = 0
        self.detection_history = deque(maxlen=100)  # Keep last 100 frame detection counts
        
        # Static detection parameters - made more conservative
        self.motion_only_mode = self.config.get('motion_only_mode', True)
        self.enable_static_detection = self.config.get('enable_static_detection', False)
        self.static_detection_interval = self.config.get('static_detection_interval', 60)  # Less frequent
        self.previous_frame = None
        self.frame_difference_threshold = self.config.get('frame_difference_threshold', 20)
        self.static_frames_buffer = deque(maxlen=5)  # Buffer to check for truly static objects
        
    def detect_birds(self, frame, mask):
        """Enhanced bird detection with motion and static detection"""
        # Motion-based detection (existing)
        motion_detections = self._detect_motion_birds(frame, mask)
        
        # Static detection (new) - balanced approach
        static_detections = []
        if (not self.motion_only_mode and self.enable_static_detection and 
            self.frame_count % self.static_detection_interval == 0 and
            len(motion_detections) < 5):  # Allow when moderate motion detections
            static_detections = self._detect_static_birds(frame)
        
        # Combine detections and remove duplicates
        all_detections = self._combine_detections(motion_detections, static_detections)
        
        # Debug output every 10 frames
        if self.frame_count % 10 == 0:
            confidences = [d['confidence'] for d in all_detections]
            max_conf = max(confidences) if confidences else 0
            avg_conf = np.mean(confidences) if confidences else 0
            # print(f"Frame {self.frame_count}: Motion={len(motion_detections)}, Static={len(static_detections)}, Combined={len(all_detections)}")
            # print(f"  Confidences: max={max_conf:.3f}, avg={avg_conf:.3f}, above_0.5={sum(1 for c in confidences if c > 0.5)}")
        
        return all_detections
    
    def _detect_motion_birds(self, frame, mask):
        """Original motion-based detection"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_contour_area < area < self.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on area and aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                confidence = self._calculate_confidence(area, aspect_ratio)
                
                if confidence > self.confidence_threshold:
                    detections.append({
                        'bbox': (x, y, w, h),
                        'confidence': confidence,
                        'centroid': (x + w//2, y + h//2),
                        'type': 'motion'
                    })
        
        return detections
    
    def _detect_static_birds(self, frame):
        """Detect static birds using conservative shape and texture analysis"""
        detections = []
        
        # Store current frame for temporal consistency check
        self.static_frames_buffer.append(frame.copy())
        
        # Only proceed if we have enough frames for comparison
        if len(self.static_frames_buffer) < 3:
            return detections
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply stronger Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Use balanced adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 5
        )
        
        # More aggressive morphological operations to reduce false positives
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Further erosion to remove small artifacts
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.erode(cleaned, erode_kernel, iterations=1)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Balanced area range for static detection
            if self.min_contour_area * 1.2 < area < self.max_contour_area * 0.9:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Strict validation for static birds
                if self._is_valid_static_bird(frame, x, y, w, h, contour):
                    # Check temporal consistency across multiple frames
                    if self._is_temporally_consistent_static(x, y, w, h):
                        aspect_ratio = w / h if h > 0 else 0
                        confidence = self._calculate_static_confidence(area, aspect_ratio, frame[y:y+h, x:x+w])
                        
                        # Balanced threshold for static detection
                        if confidence > (self.confidence_threshold * 0.9):
                            detections.append({
                                'bbox': (x, y, w, h),
                                'confidence': confidence * 0.9,  # Reduce confidence for static
                                'centroid': (x + w//2, y + h//2),
                                'type': 'static'
                            })
        
        return detections
    
    def _is_valid_static_bird(self, frame, x, y, w, h, contour):
        """Strict validation for static bird shapes to avoid shadows"""
        # Balanced size and aspect ratio checks
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:  # Less restrictive
            return False
        
        # Check if the region is significantly dark (bird-like, not shadow)
        roi = frame[y:y+h, x:x+w]
        if roi.size > 0:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            mean_intensity = np.mean(gray_roi)
            
            # Balanced darkness requirement
            frame_mean = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if mean_intensity > frame_mean * 0.8:  # Moderately darker
                return False
            
            # Check for shadow characteristics (low contrast)
            roi_std = np.std(gray_roi)
            if roi_std < 10:  # Less strict texture requirement
                return False
        
        # Balanced solidity check
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = cv2.contourArea(contour) / hull_area
            if solidity < 0.4:  # Reasonably solid
                return False
        
        # Perimeter-to-area ratio check (shadows tend to be elongated)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = 4 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter)
            if compactness < 0.2:  # Less strict compactness
                return False
        
        return True
    
    def _is_temporally_consistent_static(self, x, y, w, h):
        """Check if a static detection is consistent across multiple frames"""
        if len(self.static_frames_buffer) < 3:
            return False
        
        # Check if similar dark regions exist in previous frames
        current_center = (x + w//2, y + h//2)
        consistent_count = 0
        
        for prev_frame in list(self.static_frames_buffer)[:-1]:  # Exclude current frame
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            # Check intensity at the same location
            if (current_center[1] < gray_prev.shape[0] and 
                current_center[0] < gray_prev.shape[1]):
                
                # Sample a small region around the center
                sample_size = min(w, h, 10)
                y1 = max(0, current_center[1] - sample_size//2)
                y2 = min(gray_prev.shape[0], current_center[1] + sample_size//2)
                x1 = max(0, current_center[0] - sample_size//2)
                x2 = min(gray_prev.shape[1], current_center[0] + sample_size//2)
                
                if y2 > y1 and x2 > x1:
                    prev_intensity = np.mean(gray_prev[y1:y2, x1:x2])
                    frame_mean = np.mean(gray_prev)
                    
                    # Check if it was also dark in previous frame
                    if prev_intensity < frame_mean * 0.8:
                        consistent_count += 1
        
        # Require consistency in at least 1 out of previous frames (less strict)
        return consistent_count >= 1
    
    def _calculate_static_confidence(self, area, aspect_ratio, roi):
        """Conservative confidence calculation for static bird detection"""
        # Balanced area scoring
        area_score = max(0, 1.0 - abs(area - 140) / 140)  # Balanced optimal size
        aspect_score = max(0, 1.0 - abs(aspect_ratio - 1.0) / 2.0)  # More lenient shapes
        
        # Balanced texture and intensity analysis
        texture_score = 0.4  # Moderate default
        darkness_score = 0.4  # Moderate default
        
        if roi.size > 0:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            
            # Birds should have moderate texture variation
            texture_variation = np.std(gray_roi)
            texture_score = min(1.0, texture_variation / 20.0)  # Moderate requirement
            
            # Should be reasonably dark
            mean_intensity = np.mean(gray_roi)
            darkness_score = max(0, 1.0 - mean_intensity / 220.0)  # Less strict darkness
            
            # Penalty for very uniform regions (shadows)
            if texture_variation < 8:
                texture_score *= 0.7
        
        # More conservative weighting
        confidence = (
            area_score * 0.25 +
            aspect_score * 0.25 +
            texture_score * 0.3 +
            darkness_score * 0.2
        )
        
        # Apply moderate penalty to static detections
        return confidence * 0.85
    
    def _combine_detections(self, motion_detections, static_detections):
        """Combine motion and static detections, removing duplicates"""
        all_detections = motion_detections.copy()
        
        # Add static detections that don't overlap with motion detections
        for static_det in static_detections:
            is_duplicate = False
            static_center = static_det['centroid']
            
            for motion_det in motion_detections:
                motion_center = motion_det['centroid']
                distance = np.sqrt(
                    (static_center[0] - motion_center[0])**2 + 
                    (static_center[1] - motion_center[1])**2
                )
                
                # If static detection is close to motion detection, skip it
                if distance < 40:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_detections.append(static_det)
        
        return all_detections
    
    def _calculate_confidence(self, area, aspect_ratio):
        """Calculate confidence score based on area and aspect ratio"""
        # Normalize area (assuming birds are typically 100-300 pixels in area)
        area_score = 1.0 - abs(area - 200) / 200
        area_score = max(0, min(1, area_score))
        
        # Normalize aspect ratio (assuming birds have aspect ratio close to 1)
        aspect_score = 1.0 - abs(aspect_ratio - 1.0)
        aspect_score = max(0, min(1, aspect_score))
        
        # Combine scores
        confidence = (area_score * 0.6 + aspect_score * 0.4)
        return confidence
    
    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default parameters.")
            return {}
    
    def _is_valid_bird_shape(self, contour, area, width, height):
        """Relaxed shape validation for swiftlet detection"""
        # Basic dimension checks - more lenient for small birds
        if width < 3 or height < 3 or width > 300 or height > 300:
            return False
        
        # Aspect ratio check - more lenient for bird shapes
        aspect_ratio = width / height
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            return False
        
        # Solidity check - more lenient for birds in flight
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.15:  # More lenient for birds with wings
                return False
        
        # Extent check - more lenient
        rect_area = width * height
        extent = area / rect_area
        if extent < 0.1:  # More lenient
            return False
        
        return True
    
    def _calculate_enhanced_confidence(self, contour, area, width, height, roi):
        """Swiftlet-optimized confidence calculation"""
        # Area score - broader range for swiftlets
        if area < 100:
            area_score = 0.8  # Small birds get good score
        elif area < 200:
            area_score = 1.0  # Optimal size
        else:
            area_score = max(0.5, 1.0 - (area - 200) / 300)  # Larger birds penalized less
        
        # Aspect ratio score - more lenient
        aspect_ratio = width / height
        if 0.5 <= aspect_ratio <= 2.0:
            aspect_score = 1.0
        else:
            aspect_score = max(0.3, 1.0 - abs(aspect_ratio - 1.0) / 3.0)
        
        # Compactness score - simplified and more lenient
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = 4 * math.pi * area / (perimeter * perimeter)
            compactness_score = min(1.0, compactness * 2)  # More lenient
        else:
            compactness_score = 0.5
        
        # Simple weighted combination - focus on area and shape
        confidence = (
            area_score * 0.5 +
            aspect_score * 0.3 +
            compactness_score * 0.2
        )
        
        return max(0.1, confidence)  # Minimum confidence to avoid zero

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

    def _create_tracker(self):
        """Create tracker with compatibility fallback across OpenCV builds."""
        force_mosse_only = self.config.get('force_mosse_only', True)
        legacy = getattr(cv2, 'legacy', None)

        if force_mosse_only:
            mosse_factory = None
            if legacy is not None:
                mosse_factory = getattr(legacy, 'TrackerMOSSE_create', None)
            if mosse_factory is None:
                mosse_factory = getattr(cv2, 'TrackerMOSSE_create', None)
            if callable(mosse_factory):
                try:
                    return mosse_factory()
                except Exception:
                    return None
            return None

        factories = []
        if legacy is not None:
            factories.extend([
                getattr(legacy, 'TrackerMOSSE_create', None),
                getattr(legacy, 'TrackerKCF_create', None),
                getattr(legacy, 'TrackerCSRT_create', None),
            ])

        factories.extend([
            getattr(cv2, 'TrackerMOSSE_create', None),
            getattr(cv2, 'TrackerKCF_create', None),
            getattr(cv2, 'TrackerCSRT_create', None),
            getattr(cv2, 'TrackerMIL_create', None),
        ])

        for factory in factories:
            if callable(factory):
                try:
                    return factory()
                except Exception:
                    continue
        return None
    
    def update_trackers(self, frame, detections):
        """Simplified tracker management - like original but working"""
        self.center_box = self._get_center_box(frame.shape[1], frame.shape[0])
        updated_trackers = []
        used_detections = []
        
        # Update existing trackers (primary: detection matching by centroid distance)
        for tracker in self.trackers:
            prev_x, prev_y, prev_w, prev_h = tracker['bbox']
            prev_center = (prev_x + prev_w // 2, prev_y + prev_h // 2)
            lost_frames = tracker.get('lost_frames', 0)
            velocity = tracker.get('velocity', (0.0, 0.0))

            expected_center = prev_center
            if lost_frames > 0:
                expected_center = (
                    int(prev_center[0] + velocity[0]),
                    int(prev_center[1] + velocity[1])
                )

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
                distance = np.sqrt(
                    (expected_center[0] - det_center[0]) ** 2 +
                    (expected_center[1] - det_center[1]) ** 2
                )
                if distance < best_distance and distance < allowed_distance:
                    best_distance = distance
                    best_match = i

            if best_match is not None:
                detection = detections[best_match]
                used_detections.append(best_match)
                candidate_bbox = detection['bbox']
                if self._is_reasonable_bbox_transition(tracker['bbox'], candidate_bbox):
                    tracker['bbox'] = self._smooth_bbox(tracker['bbox'], candidate_bbox)
                else:
                    tracker['lost_frames'] = tracker.get('lost_frames', 0) + 1
                    tracker['confidence'] *= 0.95
                    if (
                        tracker['confidence'] > 0.3 and
                        tracker.get('lost_frames', 0) <= self.tracker_max_lost_frames
                    ):
                        updated_trackers.append(tracker)
                    continue
                tracker['confidence'] = detection['confidence']
                tracker['detection_type'] = detection.get('type', tracker.get('detection_type', 'motion'))
                new_center = (
                    tracker['bbox'][0] + tracker['bbox'][2] // 2,
                    tracker['bbox'][1] + tracker['bbox'][3] // 2
                )
                tracker['velocity'] = (
                    float(new_center[0] - prev_center[0]),
                    float(new_center[1] - prev_center[1])
                )
                tracker['last_update_source'] = 'detection'
                tracker['last_detection_frame'] = self.frame_count
                tracker['lost_frames'] = 0
                self._update_crossing_count(tracker, tracker['bbox'])
                updated_trackers.append(tracker)
                continue

            # Fallback: use OpenCV tracker prediction when no detection matched
            tracker_obj = tracker.get('tracker')
            success = False
            if self.use_tracker_prediction and tracker_obj is not None:
                try:
                    success, box = tracker_obj.update(frame)
                except Exception:
                    success = False

            if success:
                x, y, w, h = [int(v) for v in box]
                candidate_bbox = (x, y, w, h)
                if self._is_reasonable_bbox_transition(tracker['bbox'], candidate_bbox):
                    tracker['bbox'] = self._smooth_bbox(tracker['bbox'], candidate_bbox)
                tracker['confidence'] *= 0.95
                tracker['lost_frames'] = tracker.get('lost_frames', 0) + 1
                tracker['last_update_source'] = 'prediction'
                self._update_crossing_count(tracker, tracker['bbox'])
            else:
                tracker['lost_frames'] = tracker.get('lost_frames', 0) + 1
                vx, vy = velocity
                damping = float(self.tracker_velocity_damping)
                px = int(prev_x + vx)
                py = int(prev_y + vy)
                px = max(0, min(self.width - prev_w, px))
                py = max(0, min(self.height - prev_h, py))
                tracker['bbox'] = (px, py, prev_w, prev_h)
                tracker['velocity'] = (vx * damping, vy * damping)
                tracker['confidence'] *= 0.96
                tracker['last_update_source'] = 'prediction'
                self._update_crossing_count(tracker, tracker['bbox'])

            if (
                tracker['confidence'] > self.tracker_min_confidence_keep and
                tracker.get('lost_frames', 0) <= self.tracker_max_lost_frames
            ):
                updated_trackers.append(tracker)
        
        # Create new trackers for unmatched detections
        new_trackers_created = 0
        for i, detection in enumerate(detections):
            if i not in used_detections and detection['confidence'] > 0.5:
                # Try different tracker types
                bbox = detection['bbox']
                x, y, w, h = bbox
                
                # Validate bbox
                if w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= self.width and y + h <= self.height:
                    tracker = self._create_tracker()
                    success = True
                    if tracker is not None:
                        try:
                            success = tracker.init(frame, bbox)
                        except Exception:
                            success = False
                    
                    if success:
                        self.bird_id_counter += 1
                        new_trackers_created += 1
                        
                        updated_trackers.append({
                            'tracker': tracker,
                            'id': self.bird_id_counter,
                            'bbox': bbox,
                            'confidence': detection['confidence'],
                            'detection_type': detection.get('type', 'motion'),
                            'lost_frames': 0,
                            'inside_center_box': self._is_inside_center_box((x + w // 2, y + h // 2)),
                            'prev_center': (x + w // 2, y + h // 2),
                            'velocity': (0.0, 0.0),
                            'last_counted_time': None,
                            'last_update_source': 'detection',
                            'last_detection_frame': self.frame_count
                        })
                    else:
                        print(f"Failed to initialize MOSSE tracker for detection {i}")
                else:
                    print(f"Invalid bbox for detection {i}: {bbox}")
        
        if new_trackers_created > 0:
            print(f"Frame {self.frame_count}: Created {new_trackers_created} new trackers")
        
        self.trackers = updated_trackers
        self.bird_count = len(self.trackers)
    
    def draw_annotations(self, frame):
        """Draw bounding boxes and labels on the frame"""
        frame_height, frame_width = frame.shape[:2]
        self.center_box = self._get_center_box(frame_width, frame_height)
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
            label = f"Swiftlet: {bird_id} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            
            cv2.rectangle(frame, (x, label_y - label_size[1] - 5), 
                         (x + label_size[0] + 5, label_y + 5), color, -1)
            cv2.putText(frame, label, (x + 2, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw total count
        count_prefix = "Counter Burung Walet: "
        count_number = str(self.crossing_count)
        counter_text_thickness = int(self.config.get('counter_text_thickness', 2))
        counter_number_thickness = int(self.config.get('counter_number_thickness', 3))

        # Use slim font style consistent with debug text
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Calculate responsive text scale based on frame width
        base_width = 1920  # Reference width for scaling
        text_scale = 1.2 * (frame_width / base_width)  # Responsive scaling

        # Calculate responsive positions
        x_position = int(frame_width * 0.65)  # 65% from left
        y_position_title = int(frame_height * 0.06)  # 6% from top
        y_position_count = int(frame_height * 0.11)  # 11% from top

        # Ensure text fits within frame
        text_size_title, _ = cv2.getTextSize(self.device_name, font, text_scale, 1)
        text_size_prefix, _ = cv2.getTextSize(count_prefix, font, text_scale, 1)

        # Adjust position if text would go outside frame
        if x_position + text_size_title[0] > frame_width - 20:
            x_position = frame_width - text_size_title[0] - 20

        # Draw the location text (no outline)
        cv2.putText(frame, self.device_name, (x_position, y_position_title), 
            font, text_scale, (255, 255, 255), counter_text_thickness)

        # Draw the prefix (no outline)
        cv2.putText(frame, count_prefix, (x_position, y_position_count), 
            font, text_scale, (255, 255, 255), counter_text_thickness)

        # Calculate the position for the count number
        number_x = x_position + text_size_prefix[0]

        # Ensure number fits within frame
        text_size_number, _ = cv2.getTextSize(count_number, font, text_scale, 1)
        if number_x + text_size_number[0] > frame_width - 20:
            number_x = x_position
            y_position_count += int(text_size_prefix[1] * 1.5)  # Move to next line

        
        text_scale_number = 1.8 * (frame_width / base_width)  # Responsive scaling

        # Then draw the main count number in success green on top
        cv2.putText(frame, count_number, (number_x, y_position_count), 
            font, text_scale_number, (50, 255, 50), counter_number_thickness)


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
            
            # Store previous frame for static detection
            if self.previous_frame is not None:
                pass  # Could use for frame differencing if needed
            self.previous_frame = frame.copy()
            
            # Detect birds (motion + static)
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
        """Swiftlet-optimized mask preprocessing"""
        # Remove shadows but keep some gray areas for dark birds
        mask[mask == 127] = 200  # Convert shadows to lighter gray instead of black
        
        # Lighter morphological operations to preserve small birds
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_kernel)
        
        # Light Gaussian blur
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Lower threshold to capture darker birds
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _apply_temporal_consistency(self, detections):
        """Relaxed temporal consistency for swiftlets"""
        # Store current frame detections
        self.frame_buffer.append(detections)
        
        # For first few frames, accept all detections
        if len(self.frame_buffer) < 3:
            return detections
        
        # More lenient filtering for fast-moving birds
        consistent_detections = []
        
        for detection in detections:
            # Accept high confidence detections immediately
            if detection['confidence'] > 0.6:
                consistent_detections.append(detection)
                continue
            
            # Check for temporal consistency with larger search radius
            consistency_score = 0
            for prev_detections in list(self.frame_buffer)[:-1]:
                for prev_det in prev_detections:
                    distance = np.sqrt(
                        (detection['centroid'][0] - prev_det['centroid'][0])**2 + 
                        (detection['centroid'][1] - prev_det['centroid'][1])**2
                    )
                    if distance < 100:  # Larger search radius for fast birds
                        consistency_score += 1
                        break
            
            # Keep detection if it has consistency or reasonable confidence
            if consistency_score > 0 or detection['confidence'] > 0.4:
                consistent_detections.append(detection)
        
        return consistent_detections
    
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