"""
================================================================================
 ZENITY ROV — ai_engine.py  |  "The Brain"
 Version: 3.0 (Single-Model Master)
================================================================================
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque

# PID CONTROLLER

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, max_out: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.integral    = 0.0
        self.prev_error  = 0.0
        self._integral_limit = max_out / max(kp, 1e-6)

    def update(self, error: float) -> float:
        self.integral = max(-self._integral_limit, min(self._integral_limit, self.integral + error))
        derivative       = error - self.prev_error
        self.prev_error  = error
        raw = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return max(-self.max_out, min(self.max_out, raw))

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0

# ZENITY BRAIN

class ZenityBrain:
    # ── Tuneable constants 
    ACTION_AREA_THRESHOLD = 0.04   # Object must occupy ≥4 % of screen to trigger
    STOP_DEBOUNCE_FRAMES  = 3      
    LANE_WIDTH_ESTIMATE   = 300    
    MIN_CONTOUR_AREA      = 150    

    def __init__(self):
        # Load your future merged model here
        model_path = "yolo26s.pt" 
        print(f"[ZenityBrain] Loading {model_path} …")

        self.yolo = YOLO(model_path)

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.yolo.to(self.device)
        print(f"[ZenityBrain] Model pinned to → {self.device.upper()}")

        self.pid   = PIDController(kp=0.20, ki=0.00, kd=0.05, max_out=35.0)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._stop_buffer: deque = deque(maxlen=self.STOP_DEBOUNCE_FRAMES)

    def process_frame(self, frame_bgr: np.ndarray):
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None, False

        h, w        = frame_bgr.shape[:2]
        frame_area  = w * h
        display     = frame_bgr.copy()

        stop_raw = self._run_yolo_stage(frame_bgr, display, frame_area)

        self._stop_buffer.append(stop_raw)
        stop_detected = (len(self._stop_buffer) == self.STOP_DEBOUNCE_FRAMES and all(self._stop_buffer))

        steering_angle = self._run_lane_stage(frame_bgr, display, h, w)

        return display, steering_angle, stop_detected

    def _run_yolo_stage(self, frame_bgr, display, frame_area: int) -> bool:
        h, w = frame_bgr.shape[:2]
        stop_close = False

        try:
            results = self.yolo(frame_bgr, conf=0.45, imgsz=320, verbose=False, device=self.device)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get exact name from model
                    class_name = self.yolo.names[cls_id].upper()

                    # ── STRICT FILTERING: Ignore everything except these 6 classes ──
                    valid_targets = ["PERSON", "STOP", "TRAFFIC LIGHT", "LIGHT", "30", "40", "PARKING"]
                    if not any(target in class_name for target in valid_targets):
                        continue # Skip drawing laptops, TVs, etc.

                    box_area   = (x2 - x1) * (y2 - y1)
                    area_ratio = box_area / frame_area

                    color = (255, 100, 0) # Default Orange
                    label = f"{class_name} {conf:.2f}"

                    # ── Area Percentage Logic (Proximity Triggers) 
                    
                    if "PERSON" in class_name:
                        if area_ratio >= self.ACTION_AREA_THRESHOLD:
                            stop_close = True
                            color = (0, 0, 255)
                            label = "BRAKING — PERSON IN PATH"
                        else:
                            color = (255, 100, 0)
                            label = f"PERSON AHEAD (dist:{area_ratio:.3f})"

                    elif "STOP" in class_name:
                        if area_ratio >= self.ACTION_AREA_THRESHOLD:
                            stop_close = True
                            color = (0, 0, 255) 
                            label = "BRAKING — STOP SIGN"
                        else:
                            color = (0, 165, 255)
                            label = f"STOP AHEAD (dist:{area_ratio:.3f})"
                            
                    elif "TRAFFIC LIGHT" in class_name or "LIGHT" in class_name:
                        light_state, _ = self._classify_traffic_light(frame_bgr, x1, y1, x2, y2)
                        label = f"LIGHT:{light_state} {conf:.2f}"
                        if light_state == "RED":
                            stop_close = True
                            color = (0, 0, 255)
                            cv2.putText(display, "RED LIGHT — STOPPING", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        elif light_state == "GREEN":
                            color = (0, 255, 0)

                    elif "30" in class_name:
                        if area_ratio >= self.ACTION_AREA_THRESHOLD:
                            color = (0, 255, 255) 
                            label = "30 ZONE ENFORCED"
                            cv2.putText(display, "LIMIT: 30", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        else:
                            color = (0, 200, 200)
                            label = f"30 SPEED (dist:{area_ratio:.3f})"
                            
                    elif "40" in class_name:
                        if area_ratio >= self.ACTION_AREA_THRESHOLD:
                            color = (0, 255, 255) 
                            label = "40 ZONE ENFORCED"
                            cv2.putText(display, "LIMIT: 40", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        else:
                            color = (0, 200, 200)
                            label = f"40 SPEED (dist:{area_ratio:.3f})"
                            
                    elif "PARKING" in class_name:
                        if area_ratio >= self.ACTION_AREA_THRESHOLD:
                            color = (255, 0, 255) 
                            label = "AUTO-PARK INITIATED"
                        else:
                            color = (200, 0, 200)
                            label = f"PARKING (dist:{area_ratio:.3f})"

                    # Draw the bounding box
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, label, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        except Exception as exc:
            print(f"[ZenityBrain] YOLO error: {exc}")

        return stop_close

    def _classify_traffic_light(self, frame_bgr, x1, y1, x2, y2):
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0: return "UNKNOWN", (200, 200, 200)
        
        roi_h = roi.shape[0]
        third = max(roi_h // 3, 1)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        red_mask = cv2.bitwise_or(cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)), cv2.inRange(hsv, (160, 120, 70), (180, 255, 255)))
        green_mask = cv2.inRange(hsv, (40, 70, 70), (90, 255, 255))
        
        red_px = cv2.countNonZero(red_mask[:third, :])
        green_px = cv2.countNonZero(green_mask[2*third:, :])
        
        if red_px > green_px and red_px > 5: return "RED", (0, 0, 255)
        elif green_px > red_px and green_px > 5: return "GREEN", (0, 255, 0)
        else: return "UNKNOWN", (200, 200, 200)

    def _run_lane_stage(self, frame_bgr, display, h: int, w: int):
        try:
            roi_y = int(h * 0.50)
            roi = frame_bgr[roi_y:h, :]
            roi_h, roi_w = roi.shape[:2]
            mid_x = roi_w // 2

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = self.clahe.apply(gray)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)

            left_cx = self._find_lane_centroid(mask[:, :mid_x], 0)
            right_cx = self._find_lane_centroid(mask[:, mid_x:], mid_x)

            if left_cx is not None: cv2.circle(display, (left_cx, roi_y + roi_h // 2), 8, (255, 0, 0), -1)
            if right_cx is not None: cv2.circle(display, (right_cx, roi_y + roi_h // 2), 8, (255, 0, 255), -1)

            center_x = roi_w // 2
            half_lane = self.LANE_WIDTH_ESTIMATE // 2

            if left_cx is not None and right_cx is not None:
                target_x = (left_cx + right_cx) // 2
                cv2.line(display, (left_cx, h - 20), (right_cx, h - 20), (0, 255, 255), 2)
            elif left_cx is not None:
                target_x = left_cx + half_lane
            elif right_cx is not None:
                target_x = right_cx - half_lane
            else:
                self.pid.reset()
                return None

            error = target_x - center_x
            steering_angle = self.pid.update(error)
            cv2.line(display, (center_x, h), (target_x, roi_y + 50), (0, 255, 0), 3)
            return steering_angle
        except Exception as exc:
            self.pid.reset()
            return None

    def _find_lane_centroid(self, mask_half: np.ndarray, offset_x: int):
        contours, _ = cv2.findContours(mask_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.MIN_CONTOUR_AREA: return None
        M = cv2.moments(largest)
        if M["m00"] == 0: return None
        return int(M["m10"] / M["m00"]) + offset_x