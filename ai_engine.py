"""
================================================================================
 ZENITY ROV — ai_engine.py  |  "The Brain"
 Version: 2.0 (Production-Grade)
================================================================================
 Responsibilities:
   - YOLO26 Small multi-object detection (Person, Car, Traffic Light, Stop Sign)
   - Dual-lane detection pipeline (ROI → CLAHE → Adaptive Threshold → Contours)
   - PID steering controller
   - Distance estimation via bounding-box area ratio
   - Traffic light color classification (Red / Green pixel thresholding)
   - Debounced stop-sign state (prevents flickering brake commands)
================================================================================
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque


# PID CONTROLLER

class PIDController:
    """
    Classic discrete PID with output clamping and integral wind-up guard.

    Tuning cheat-sheet for Zenity ROV:
      kp  →  higher = more aggressive turn response
      ki  →  higher = corrects long-term drift, but can cause oscillation
      kd  →  higher = dampens overshoot / jerky steering
    """

    def __init__(self, kp: float, ki: float, kd: float, max_out: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out

        self.integral    = 0.0
        self.prev_error  = 0.0

        # Anti-windup: clamp integral so it never grows unbounded
        self._integral_limit = max_out / max(kp, 1e-6)

    def update(self, error: float) -> float:
        self.integral = max(
            -self._integral_limit,
            min(self._integral_limit, self.integral + error)
        )
        derivative       = error - self.prev_error
        self.prev_error  = error

        raw = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return max(-self.max_out, min(self.max_out, raw))

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0

# ZENITY BRAIN

class ZenityBrain:
    """
    The full AI + CV perception pipeline.

    process_frame() returns:
        display (np.ndarray)   — annotated BGR frame for the dashboard
        steering_angle (float|None) — signed degrees; negative = left, positive = right
        stop_detected  (bool)  — True only when a debounced stop condition is confirmed
    """

    # ── YOLO class IDs we care about ──────────────────────────────────────────
    _TRACKED_CLASSES = [0, 2, 9, 11]          # person, car, traffic light, stop sign
    _CLASS_NAMES     = {
        0:  "PERSON",
        2:  "CAR",
        9:  "LIGHT",
        11: "STOP SIGN",
    }

    # ── Tuneable constants ────────────────────────────────────────────────────
    BRAKE_AREA_THRESHOLD  = 0.04   # stop sign occupies ≥4 % of screen → brake
    STOP_DEBOUNCE_FRAMES  = 3      # consecutive positive detections needed
    LANE_WIDTH_ESTIMATE   = 300    # pixels — used when only one line is found
    MIN_CONTOUR_AREA      = 150    # ignore tiny noise contours

    def __init__(self):
        # ── Model loading ─────────────────────────────────────────────────────
        # Use yolo26s.pt (Nano) for lowest latency on M3.
        # Swap to yolo26s.pt only if you have the M4 16 GB machine.
        model_path = "yolo26s.pt"
        print(f"[ZenityBrain] Loading {model_path} …")

        self.yolo = YOLO(model_path)

        # Pin the model to MPS (Apple Silicon GPU).
        # If MPS is unavailable (e.g., running on the Vivobook later), fall
        # back to CUDA, then CPU — without crashing.
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.yolo.to(self.device)
        print(f"[ZenityBrain] Model pinned to → {self.device.upper()}")

        # ── Sub-systems ───────────────────────────────────────────────────────
        self.pid   = PIDController(kp=0.20, ki=0.00, kd=0.05, max_out=35.0)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Debounce buffer: track last N stop-sign evaluations
        self._stop_buffer: deque = deque(maxlen=self.STOP_DEBOUNCE_FRAMES)

        print("[ZenityBrain] Warm-up pass …")
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self._run_yolo(dummy)  # force model JIT compile before real use
        print("[ZenityBrain] Ready ✓")

    # PUBLIC API

    def process_frame(self, frame_bgr: np.ndarray):
        """Main entry point called by main_rov.py on every new camera frame."""

        # Sanity check — corrupted/empty frames arrive over flaky Wi-Fi
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None, False

        h, w        = frame_bgr.shape[:2]
        frame_area  = w * h
        display     = frame_bgr.copy()

        stop_raw        = False   # raw detection this frame
        steering_angle  = None

        # ── Stage 1: YOLO multi-object detection
        stop_raw = self._run_yolo_stage(frame_bgr, display, frame_area)

        # Debounce: only declare stop if last N frames all agreed
        self._stop_buffer.append(stop_raw)
        stop_detected = (len(self._stop_buffer) == self.STOP_DEBOUNCE_FRAMES
                         and all(self._stop_buffer))

        # ── Stage 2: Dual-lane detection + PID steering 
        steering_angle = self._run_lane_stage(frame_bgr, display, h, w)

        return display, steering_angle, stop_detected

    # PRIVATE — YOLO STAGE

    def _run_yolo(self, frame):
        """Raw YOLO call — isolated so warm-up can call it separately."""
        return self.yolo(
            frame,
            conf     = 0.45,        
            imgsz    = 320,
            verbose  = False,
            device   = self.device,  
        )

    def _run_yolo_stage(self, frame_bgr, display, frame_area: int) -> bool:
        """
        Runs detection, draws boxes, classifies traffic lights, and returns
        whether a stop sign is close enough to trigger a brake.
        """
        h, w = frame_bgr.shape[:2]
        stop_close = False

        try:
            results = self._run_yolo(frame_bgr)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    box_area   = (x2 - x1) * (y2 - y1)
                    area_ratio = box_area / frame_area

                    # ── Traffic light: classify Red / Green via pixel mask ────
                    if cls_id == 9:
                        light_state, color = self._classify_traffic_light(
                            frame_bgr, x1, y1, x2, y2
                        )
                        label = f"LIGHT:{light_state} {conf:.2f}"
                        if light_state == "RED":
                            stop_close = True
                            cv2.putText(display, "RED LIGHT — STOPPING",
                                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.9, (0, 0, 255), 2)
                    # ── Stop sign: brake on proximity ────────────────────────
                    elif cls_id == 11:
                        if area_ratio >= self.BRAKE_AREA_THRESHOLD:
                            stop_close = True
                            color = (0, 0, 255)
                            label = f"STOP SIGN {conf:.2f} [BRAKE]"
                            cv2.putText(display, "BRAKING — STOP SIGN CLOSE",
                                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.9, color, 2)
                        else:
                            color = (0, 165, 255)
                            label = f"STOP SIGN {conf:.2f} (dist:{area_ratio:.3f})"
                    else:
                        color = (255, 100, 0) # Default orange box
                        # Use YOLO's internal dictionary to get the real name
                        object_name = self.yolo.names[cls_id].upper()
                        label = f"{object_name} {conf:.2f}"

                    # cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, label, (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        except Exception as exc:
            print(f"[ZenityBrain] YOLO error: {exc}")

        return stop_close

    # PRIVATE — TRAFFIC LIGHT CLASSIFIER

    def _classify_traffic_light(self, frame_bgr, x1, y1, x2, y2):
        """
        Splits the bounding box into top-third (red) and bottom-third (green)
        regions and counts saturated pixels in HSV colour space.

        Returns: (state_string, bgr_color_for_overlay)
        """
        # Guard against boxes that go out of frame
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)

        roi       = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return "UNKNOWN", (200, 200, 200)

        roi_h     = roi.shape[0]
        third     = max(roi_h // 3, 1)
        hsv       = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red spans two hue ranges in HSV (0-10 and 160-180)
        red_mask  = cv2.bitwise_or(
            cv2.inRange(hsv, (0,   120, 70), (10,  255, 255)),
            cv2.inRange(hsv, (160, 120, 70), (180, 255, 255)),
        )
        green_mask = cv2.inRange(hsv, (40, 70, 70), (90, 255, 255))

        red_px   = cv2.countNonZero(red_mask[:third,  :])
        green_px = cv2.countNonZero(green_mask[2*third:, :])

        if red_px > green_px and red_px > 5:
            return "RED",   (0, 0, 255)
        elif green_px > red_px and green_px > 5:
            return "GREEN", (0, 255, 0)
        else:
            return "UNKNOWN", (200, 200, 200)

    # PRIVATE — LANE DETECTION STAGE

    # PRIVATE — LANE DETECTION STAGE (UPGRADED FOR SHARP TURNS)

    def _run_lane_stage(self, frame_bgr, display, h: int, w: int):
        """
        Dual-lane detection upgraded for wide-angle lenses and sharp turns.
        Uses pure-white thresholding and margin masking to ignore the floor.
        """
        try:
            # 1. Look slightly higher up to see turns earlier (40% down instead of 50%)
            roi_y = int(h * 0.40)
            roi = frame_bgr[roi_y:h, :]
            roi_h, roi_w = roi.shape[:2]
            mid_x = roi_w // 2

            # 2. Add "Blindfolds" to ignore the wide-angle floor (15% on each side)
            margin = int(roi_w * 0.15) 

            # 3. Pure White Isolation (Road is Black, Lanes are White)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # ONLY keep very bright white pixels (Adjust '180' up or down if needed)
            _, mask = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)

            # Apply the blindfolds to the mask (black out the peripheral floor)
            mask[:, :margin] = 0
            mask[:, roi_w - margin:] = 0

            # 4. Find the lanes dynamically
            left_cx = self._find_lane_centroid(mask[:, :mid_x], offset_x=0)
            right_cx = self._find_lane_centroid(mask[:, mid_x:], offset_x=mid_x)

            # ── Visual feedback (Draw the ROI bounds on the dashboard) ──
            cv2.line(display, (margin, roi_y), (margin, h), (0, 0, 150), 2) # Left blindfold
            cv2.line(display, (roi_w - margin, roi_y), (roi_w - margin, h), (0, 0, 150), 2) # Right blindfold
            cv2.line(display, (0, roi_y), (w, roi_y), (150, 0, 0), 2) # Horizon line

            if left_cx is not None:
                cv2.circle(display, (left_cx, roi_y + roi_h // 2), 8, (255, 0, 0), -1)
            if right_cx is not None:
                cv2.circle(display, (right_cx, roi_y + roi_h // 2), 8, (255, 0, 255), -1)

            # ── Steering target resolution 
            center_x = roi_w // 2
            half_lane = self.LANE_WIDTH_ESTIMATE // 2

            if left_cx is not None and right_cx is not None:
                # Both lanes visible
                target_x = (left_cx + right_cx) // 2
                cv2.line(display, (left_cx, h - 20), (right_cx, h - 20), (0, 255, 255), 2)

            elif left_cx is not None:
                # Turn so sharp we lost the right lane! Follow the left curve.
                target_x = left_cx + half_lane      
                cv2.putText(display, "TRACKING LEFT ONLY", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

            elif right_cx is not None:
                # Turn so sharp we lost the left lane! Follow the right curve.
                target_x = right_cx - half_lane     
                cv2.putText(display, "TRACKING RIGHT ONLY", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

            else:
                self.pid.reset()
                cv2.putText(display, "⚠ TOTAL LANE LOSS", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                return None

            error = target_x - center_x
            steering_angle = self.pid.update(error)

            # Draw the steering intention line
            cv2.line(display, (center_x, h), (target_x, roi_y + 50), (0, 255, 0), 3)

            return steering_angle

        except Exception as exc:
            print(f"[ZenityBrain] Lane error: {exc}")
            self.pid.reset()
            return None

    def _find_lane_centroid(self, mask_half: np.ndarray, offset_x: int):
        """
        Finds the largest contour in a half-mask and returns its x centroid,
        shifted by offset_x. Returns None if no valid contour exists.
        """
        contours, _ = cv2.findContours(
            mask_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.MIN_CONTOUR_AREA:
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        return int(M["m10"] / M["m00"]) + offset_x