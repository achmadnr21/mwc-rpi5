import cv2
import numpy as np
import time
import json
from pathlib import Path
from collections import deque
import math

class SwiftletCounter:
    def __init__(self, input_video_path=None, output_video_path=None, config_path="config.json", streaming_mode=False, device_name="RBW Lantai 1"):
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
        self.max_birds_per_frame = 20  # Prevent tracker explosion
        
        # Detection parameters - back to working values
        self.min_contour_area = self.config.get('min_contour_area', 50)
        self.max_contour_area = self.config.get('max_contour_area', 500)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.tracker_max_distance = self.config.get('tracker_max_distance', 70)
        self.tracker_max_lost_frames = self.config.get('tracker_max_lost_frames', 15)
        
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
        self.enable_static_detection = self.config.get('enable_static_detection', True)
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
        if (self.enable_static_detection and 
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
    
    def update_trackers(self, frame, detections):
        """Simplified tracker management - like original but working"""
        updated_trackers = []
        used_detections = []
        
        # Update existing trackers
        for tracker in self.trackers:
            success, box = tracker['tracker'].update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in box]
                center = (x + w//2, y + h//2)
                
                # Find matching detection
                best_match = None
                best_distance = float('inf')
                
                for i, detection in enumerate(detections):
                    if i not in used_detections:
                        det_center = detection['centroid']
                        distance = np.sqrt((center[0] - det_center[0])**2 + 
                                         (center[1] - det_center[1])**2)
                        
                        if distance < best_distance and distance < 50:
                            best_distance = distance
                            best_match = i
                
                if best_match is not None:
                    used_detections.append(best_match)
                    detection = detections[best_match]
                    tracker['bbox'] = detection['bbox']
                    tracker['confidence'] = detection['confidence']
                else:
                    tracker['bbox'] = (x, y, w, h)
                    tracker['confidence'] *= 0.95  # Decay confidence
                
                if tracker['confidence'] > 0.4:  # Keep tracker if confidence is high enough
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
                    # Try MOSSE tracker (faster and more reliable)
                    tracker = cv2.legacy.TrackerMOSSE_create()
                    success = tracker.init(frame, bbox)
                    
                    if success:
                        self.bird_id_counter += 1
                        new_trackers_created += 1
                        
                        updated_trackers.append({
                            'tracker': tracker,
                            'id': self.bird_id_counter,
                            'bbox': bbox,
                            'confidence': detection['confidence'],
                            'detection_type': detection.get('type', 'motion')
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
        for tracker in self.trackers:
            x, y, w, h = tracker['bbox']
            confidence = tracker['confidence']
            bird_id = tracker['id']
            
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
            label = f"Swiftlet ({detection_type}): {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            
            cv2.rectangle(frame, (x, label_y - label_size[1] - 5), 
                         (x + label_size[0] + 5, label_y + 5), color, -1)
            cv2.putText(frame, label, (x + 2, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Get frame dimensions for responsive positioning
        frame_height, frame_width = frame.shape[:2]

        # Draw total count
        count_prefix = "Total Burung Walet: "
        count_number = str(self.bird_count)

        # Use elegant font
        font = cv2.FONT_HERSHEY_DUPLEX

        # Calculate responsive text scale based on frame width
        base_width = 1920  # Reference width for scaling
        text_scale = 1.2 * (frame_width / base_width)  # Responsive scaling

        # Calculate responsive positions
        x_position = int(frame_width * 0.65)  # 65% from left
        y_position_title = int(frame_height * 0.06)  # 6% from top
        y_position_count = int(frame_height * 0.11)  # 11% from top

        # Ensure text fits within frame
        text_size_title, _ = cv2.getTextSize(self.device_name, font, text_scale, 2)
        text_size_prefix, _ = cv2.getTextSize(count_prefix, font, text_scale, 2)

        # Adjust position if text would go outside frame
        if x_position + text_size_title[0] > frame_width - 20:
            x_position = frame_width - text_size_title[0] - 20

        # Draw the location text (no outline)
        cv2.putText(frame, self.device_name, (x_position, y_position_title), 
                font, text_scale, (255, 255, 255), 2)

        # Draw the prefix (no outline)
        cv2.putText(frame, count_prefix, (x_position, y_position_count), 
                font, text_scale, (255, 255, 255), 2)

        # Calculate the position for the count number
        number_x = x_position + text_size_prefix[0]

        # Ensure number fits within frame
        text_size_number, _ = cv2.getTextSize(count_number, font, text_scale, 2)
        if number_x + text_size_number[0] > frame_width - 20:
            number_x = x_position
            y_position_count += int(text_size_prefix[1] * 1.5)  # Move to next line

        # # Draw ONLY the count number with white outline
        # # First draw the white outline
        # for dx in [-2, -1, 0, 1, 2]:
        #     for dy in [-2, -1, 0, 1, 2]:
        #         if dx != 0 or dy != 0:
        #             cv2.putText(frame, count_number, (number_x + dx, y_position_count + dy), 
        #                     font, text_scale, (255, 255, 255), 3)

        text_scale_number = 1.8 * (frame_width / base_width)  # Responsive scaling

        # Then draw the main count number in success green on top
        cv2.putText(frame, count_number, (number_x, y_position_count), 
                font, text_scale_number, (50, 255, 50), 2)

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