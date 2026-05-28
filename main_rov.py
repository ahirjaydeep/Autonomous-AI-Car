
# =============================================================================
# AUTHOR'S NOTE
# =============================================================================
# This project, system design, logic flow, implementation decisions, testing,
# tuning, and integration were developed by the author.
#
# AI-assisted tools were used to help generate parts of the code structure,
# comments, formatting, documentation, and refactoring during development.
#
# Final architecture, engineering decisions, debugging, validation, and
# production integration were performed and reviewed by the author.
# =============================================================================



"""
================================================================================
 ZENITY ROV — ai_engine8.py  |  "The Brain"
 Version: 8.0 (Production — Camera Offset, Nearest-Pair Tracking, Lane Memory)
================================================================================
 KEY IMPROVEMENTS over v7.1:

 CAMERA OFFSET:
   - Corrects for phone camera mounted 6cm left of car centreline
   - Configurable CAMERA_OFFSET_PX constant (default 55px)

 NEAREST-PAIR LANE TRACKING:
   - Replaces v7.1's LEFT/RIGHT split (which broke on turns)
   - Finds ALL white lines in the full mask, picks the pair whose
     midpoint is closest to the car centre → always tracks correct lane
   - LANE_WIDTH_MAX = 500 rejects lane pairs that are too wide
     (prevents tracking left_boundary + right_boundary as one "lane")

 LANE MEMORY:
   - Remembers last known position of left and right lane lines (EMA)
   - On subsequent frames, matches detected lines to remembered positions
   - If a new line is too far from its remembered position (LINE_JUMP_THRESH),
     it's treated as a DIFFERENT line (e.g., outer road boundary) and ignored
   - Prevents the car from "jumping" to track the wrong lane on turns

 YOLO DETECTION:
   - Same as v7.0 (unchanged)
================================================================================
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque
import time


# ──────────────────────────────────────────────────────────────────────────────
# PID CONTROLLER
# ──────────────────────────────────────────────────────────────────────────────
# A classic Proportional-Integral-Derivative controller used to convert
# lateral pixel error (distance from lane centre) into a smooth steering angle.
#
# How it works:
#   P (proportional) → reacts to current error (bigger error = harder turn)
#   I (integral)     → reacts to accumulated past error (drift correction)
#   D (derivative)   → reacts to rate of change (dampens oscillation)
#
# Anti-windup: integral is clamped to prevent runaway values when the car
# is stuck or lanes are lost for extended periods.
# ──────────────────────────────────────────────────────────────────────────────

class PIDController:
    """
    Discrete PID controller with anti-windup clamping.

    Tuning guide for Zenity ROV (4WD skid-steer):
      kp = 0.22  → Aggressive enough to start turning at ~5px error.
                    Increase if the car is sluggish on turns.
                    Decrease if the car oscillates (wiggles side to side).
      ki = 0.0   → Disabled. Integral is risky on a car that can lose lanes
                    temporarily. Enable only if the car has persistent drift.
      kd = 0.05  → Dampens overshoot when the car snaps back from a turn.
                    Increase if the car overshoots past centre after curves.
                    Decrease if the car feels too sluggish entering turns.
                    (Reduced from 0.08 to prevent startup derivative spike.)
      max_out = 35.0 → Maximum steering angle in degrees. The tank-drive
                       mixer maps ±35° to full differential.
    """

    def __init__(self, kp: float, ki: float, kd: float, max_out: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out

        self.integral = 0.0
        self.prev_error = 0.0

        # Anti-windup: integral can never exceed this value
        # (proportional component alone at this integral would hit max_out)
        self._integral_limit = max_out / max(kp, 1e-6)

    def update(self, error: float) -> float:
        """
        Feed a new error value and get the PID output.

        Args:
            error: signed pixel offset. Positive = car is LEFT of target
                   (needs to steer right). Negative = car is RIGHT (steer left).

        Returns:
            Steering angle in degrees, clamped to [-max_out, +max_out].
        """
        # I — accumulate error with clamping
        self.integral = max(
            -self._integral_limit,
            min(self._integral_limit, self.integral + error)
        )

        # D — rate of change of error
        derivative = error - self.prev_error
        self.prev_error = error

        # PID formula
        raw = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Clamp output to safe steering range
        return max(-self.max_out, min(self.max_out, raw))

    def reset(self):
        """Reset integrator and derivative memory. Called on total lane loss."""
        self.integral = 0.0
        self.prev_error = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# ZENITY BRAIN  v7.0
# ──────────────────────────────────────────────────────────────────────────────
# This is the main class that processes every camera frame through two
# parallel pipelines:
#   1) YOLO object detection → identifies stop signs, people, lights, etc.
#   2) Lane detection → finds lane lines, computes steering angle via PID
#
# The constructor loads the YOLO model, initialises the PID controller,
# and performs a warm-up inference to JIT-compile the model.
# ──────────────────────────────────────────────────────────────────────────────

class ZenityBrain:
    """
    AI perception engine for Zenity ROV.

    process_frame() is the single public API called by main_rov7.py on every
    new camera frame. It returns:
        display        (np.ndarray)  — annotated BGR frame for the dashboard
        steering_angle (float|None)  — signed degrees; negative=left, positive=right
        stop_detected  (bool)        — True when debounced stop condition confirmed
        detections     (dict)        — metadata: {'speed_limit': int|None, 'person': bool, ...}
    """

    # ══════════════════════════════════════════════════════════════════════════
    # TUNEABLE CONSTANTS  (v8.0 — Camera Offset + Nearest-Pair Tracking)
    # ══════════════════════════════════════════════════════════════════════════
    # ADJUST THESE FIRST before changing any logic.

    # --- CAMERA OFFSET ---
    # The phone camera is NOT centred on the chassis. It is mounted ~6cm LEFT
    # of the car's centreline. In the image, the car's true centre appears
    # shifted RIGHT of the image centre.
    # This offset corrects for that: cx_img = image_centre + CAMERA_OFFSET_PX.
    #
    # HOW TO CALIBRATE:
    #   1. Place car perfectly centred on a straight lane
    #   2. Run the AI, look at the dashboard
    #   3. If the green arrow points RIGHT → increase this value
    #   4. If the green arrow points LEFT  → decrease this value
    #   5. When the arrow is near-zero on a straight lane, you're done
    #
    # Rough estimate: ~9 px/cm at typical phone height (25-30cm).
    #   6 cm × 9 px/cm ≈ 55 pixels.
    CAMERA_OFFSET_PX = 0    # positive = camera is LEFT of car centre
                              # ↑ increase if car still drifts left
                              # ↓ decrease if car drifts right
                              # Set to 0 if camera IS centred

    # --- WHITE LINE DETECTION ---
    WHITE_THRESH = 180       # Fixed threshold for white-on-black tracks
                              # ↑ higher (200) = only catches bright white
                              # ↓ lower (150) = catches dimmer lines, more noise

    # --- ROI (Region of Interest) ---
    ROI_TOP_FRACTION = 0.45  # Bottom 55% of frame (near-road focus)

    # --- LANE GEOMETRY ---
    LANE_WIDTH_INIT = 260    # Initial guess for lane width in pixels
    LANE_WIDTH_MIN  = 80     # Reject impossibly narrow line pairs
    LANE_WIDTH_MAX  = 500    # Reject impossibly wide pairs (prevents tracking
                              # left_boundary + right_boundary as one lane)

    # --- CONTOUR NOISE GATE ---
    MIN_CONTOUR_AREA = 300   # Reject white blobs smaller than this (px²)

    # --- STEERING DEADBAND ---
    STEERING_DEADBAND_PX = 12  # ±12px error → zero steering (drive straight)

    # --- LANE MEMORY ---
    # Maximum distance (px) a lane line can jump between frames.
    # If a detected line is further than this from its remembered position,
    # it's treated as a DIFFERENT line (probably the outer road boundary)
    # and ignored. This prevents the car from jumping to track the wrong
    # lane when the inner line leaves the frame on turns.
    LINE_JUMP_THRESH = 120   # ↑ higher = more tolerant of fast movement
                              # ↓ lower = stricter continuity, may lose lines

    # --- YOLO / BRAKING ---
    BRAKE_AREA_THRESHOLD = 0.015
    STOP_DEBOUNCE_FRAMES = 3


    def __init__(self):
        # ── Model Loading ─────────────────────────────────────────────────────
        # Uses yolo26s.pt — the Small model. Good balance of speed and accuracy.
        # Swap to yolo26n.pt (Nano) for faster but less accurate inference.
        # Swap to yolo11s.pt or a custom zenity_master.pt if you have one.
        model_path = "best.pt"
        print(f"[ZenityBrain] Loading {model_path} …")

        self.yolo = YOLO(model_path)

        # Pin model to the fastest available accelerator.
        # MPS = Apple Silicon GPU (M1/M2/M3/M4) — fastest on Mac.
        # CUDA = NVIDIA GPU — use if running on a Linux/Windows GPU machine.
        # CPU = fallback — works everywhere but slow.
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.yolo.to(self.device)
        print(f"[ZenityBrain] Model pinned to → {self.device.upper()}")

        # ── Sub-systems ───────────────────────────────────────────────────────
        # PID controller — converts pixel error into steering angle
        self.pid = PIDController(kp=0.22, ki=0.0, kd=0.05, max_out=35.0)

        # CLAHE — Contrast-Limited Adaptive Histogram Equalisation
        # Normalises lighting across the frame so lane lines are consistent
        # brightness whether they're in shadow or direct light.
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # Stop-sign debounce buffer: tracks last N detection results
        self._stop_buffer: deque = deque(maxlen=self.STOP_DEBOUNCE_FRAMES)

        # Auto-calibrated lane width (updated whenever both lines are visible)
        self._lane_width = self.LANE_WIDTH_INIT

        # ── Lane memory ───────────────────────────────────────────────────────
        # Remember the last known x-position of each lane line.
        # Used to prevent the car from jumping to track the wrong line when
        # the inner line leaves the frame on turns.
        self._last_left_x  = None  # EMA-smoothed left line position
        self._last_right_x = None  # EMA-smoothed right line position

        # ── EMA steering smoother ─────────────────────────────────────────────
        self._steer_ema = 0.0
        self._steer_ema_alpha = 0.6  # 0.6 new + 0.4 old

        # ── Startup frame skip ────────────────────────────────────────────────
        self._startup_skip_frames = 3
        self._frame_count = 0

        # ── Warm-up ───────────────────────────────────────────────────────────
        # Run a dummy inference to force PyTorch to JIT-compile the model.
        # Without this, the first real frame takes 2-5 seconds.
        print("[ZenityBrain] Warm-up inference …")
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self._run_yolo(dummy)
        print("[ZenityBrain] Ready ✓\n")


    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Main entry point. Called by main_rov7.py on every new camera frame.

        Args:
            frame_bgr: Raw BGR image from the camera (any resolution).

        Returns:
            display        — Annotated BGR frame with bounding boxes, lane markers, HUD.
            steering_angle — Signed steering angle in degrees. None if lanes are totally lost.
            stop_detected  — True if a debounced stop condition is confirmed.
            detections     — Dict with metadata:
                             {
                               'speed_limit': int|None,  # 30 or 40 if detected, else None
                               'person_close': bool,     # True if person within braking distance
                               'stop_sign_close': bool,  # True if stop sign within braking distance
                               'red_light': bool,        # True if red traffic light detected
                             }
        """
        # ── Sanity check ─────────────────────────────────────────────────────
        # Corrupted or empty frames arrive over flaky Wi-Fi connections.
        if frame_bgr is None or frame_bgr.size == 0:
            print("[ZenityBrain] ⚠ Received empty/corrupted frame — skipping")
            return None, None, False, {}

        h, w = frame_bgr.shape[:2]
        frame_area = w * h
        display = frame_bgr.copy()

        # ── Stage 1: YOLO Object Detection ────────────────────────────────────
        # Detects stop signs, people, traffic lights, speed signs, parking signs.
        # Returns raw stop flag + metadata dict.
        stop_raw, detections = self._run_yolo_stage(frame_bgr, display, frame_area)

        # Debounce: require N consecutive positive frames before confirming stop.
        # This prevents a single flickering detection from causing a brake event.
        self._stop_buffer.append(stop_raw)
        stop_detected = (
            len(self._stop_buffer) == self.STOP_DEBOUNCE_FRAMES
            and all(self._stop_buffer)
        )

        # ── Stage 2: Lane Detection + PID Steering ────────────────────────────
        # Finds lane lines using multi-slice scanning, computes steering angle.
        steering_angle = self._run_lane_stage(frame_bgr, display, h, w)

        return display, steering_angle, stop_detected, detections


    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE — YOLO STAGE
    # ══════════════════════════════════════════════════════════════════════════

    def _run_yolo(self, frame):
        """
        Raw YOLO inference call. Isolated so warm-up can call it without
        the full detection pipeline.

        Uses imgsz=320 for speed. Increase to 640 for more accuracy but
        ~4x slower inference (not recommended for real-time driving).
        """
        return self.yolo(
            frame,
            conf=0.45,        # confidence threshold — objects below 45% are ignored
            imgsz=320,        # input resolution — 320px is the speed/accuracy sweet spot
            verbose=False,    # suppress YOLO's own console output
            device=self.device,
        )

    def _run_yolo_stage(self, frame_bgr, display, frame_area: int):
        """
        Runs YOLO detection, draws bounding boxes, classifies traffic lights,
        and determines if any stop-triggering object is close enough.

        STRICT CLASS FILTERING: Only processes Person, Stop, Traffic Light,
        30, 40, and Parking. All other YOLO classes (laptop, TV, etc.) are
        silently ignored — they're false positives from the COCO pretrained model.

        Returns:
            stop_close (bool) — raw (not debounced) stop trigger
            detections (dict) — metadata about what was detected
        """
        stop_close = False
        detections = {
            'speed_limit': None,      # will be set to 30 or 40 if detected
            'person_close': False,
            'stop_sign_close': False,
            'red_light': False,
        }

        try:
            results = self._run_yolo(frame_bgr)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get the class name from the model's dictionary
                    class_name = self.yolo.names[cls_id].upper()

                    # ── STRICT FILTER: Only process our target classes ────────
                    # This list must match your custom-trained model's classes.
                    # For COCO-pretrained models, "PERSON" and "STOP SIGN" work.
                    # For custom models, add your exact class names here.
                    valid_targets = [
                        "PERSON", "STOP", "TRAFFIC LIGHT", "LIGHT",
                        "30", "40", "PARKING"
                    ]
                    if not any(target in class_name for target in valid_targets):
                        continue  # skip irrelevant classes (laptop, TV, etc.)

                    box_area = (x2 - x1) * (y2 - y1)
                    area_ratio = box_area / frame_area

                    color = (255, 100, 0)  # default: orange
                    label = f"{class_name} {conf:.2f}"

                    # ── PERSON — Emergency Stop ──────────────────────────────
                    # A person within braking distance triggers immediate stop.
                    # This is the highest-priority AI detection (above stop signs).
                    if "PERSON" in class_name:
                        if area_ratio >= self.BRAKE_AREA_THRESHOLD:
                            stop_close = True
                            detections['person_close'] = True
                            color = (0, 0, 255)  # red box
                            label = "⚠ BRAKING — PERSON IN PATH"
                            cv2.putText(
                                display, "BRAKING — PERSON IN PATH",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
                            )
                        else:
                            color = (255, 100, 0)  # orange — detected but far
                            label = f"PERSON AHEAD (d:{area_ratio:.3f})"

                    # ── STOP SIGN — Brake on Proximity ───────────────────────
                    # Only triggers when the sign is large enough in frame (close).
                    # The debounce + cooldown logic is in main_rov7.py.
                    elif "STOP" in class_name:
                        if area_ratio >= self.BRAKE_AREA_THRESHOLD:
                            stop_close = True
                            detections['stop_sign_close'] = True
                            color = (0, 0, 255)
                            label = "⚠ BRAKING — STOP SIGN"
                            cv2.putText(
                                display, "BRAKING — STOP SIGN CLOSE",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                            )
                        else:
                            color = (0, 165, 255)  # orange — seen but far
                            label = f"STOP AHEAD (d:{area_ratio:.3f})"

                    # ── TRAFFIC LIGHT — HSV Color Classifier ─────────────────
                    # The bounding box is passed to a separate classifier that
                    # counts red vs green pixels in HSV space.
                    elif "TRAFFIC LIGHT" in class_name or "LIGHT" in class_name:
                        light_state, _ = self._classify_traffic_light(
                            frame_bgr, x1, y1, x2, y2
                        )
                        label = f"LIGHT:{light_state} {conf:.2f}"
                        if light_state == "RED":
                            stop_close = True
                            detections['red_light'] = True
                            color = (0, 0, 255)
                            cv2.putText(
                                display, "RED LIGHT — STOPPING",
                                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                            )
                        elif light_state == "GREEN":
                            color = (0, 255, 0)

                    # ── SPEED 30 — Speed Zone Enforcement ────────────────────
                    # When close enough, reports speed_limit=30 to main_rov7.py.
                    # main_rov7.py will reduce BASE_SPEED accordingly.
                    elif "30" in class_name:
                        if area_ratio >= self.BRAKE_AREA_THRESHOLD:
                            detections['speed_limit'] = 30
                            color = (0, 255, 255)  # cyan
                            label = "30 ZONE ENFORCED"
                            cv2.putText(
                                display, "SPEED LIMIT: 30",
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                            )
                        else:
                            color = (0, 200, 200)
                            label = f"30 SPEED (d:{area_ratio:.3f})"

                    # ── SPEED 40 — Speed Zone Enforcement ────────────────────
                    elif "40" in class_name:
                        if area_ratio >= self.BRAKE_AREA_THRESHOLD:
                            detections['speed_limit'] = 40
                            color = (0, 255, 255)
                            label = "40 ZONE ENFORCED"
                            cv2.putText(
                                display, "SPEED LIMIT: 40",
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                            )
                        else:
                            color = (0, 200, 200)
                            label = f"40 SPEED (d:{area_ratio:.3f})"

                    # ── PARKING — Auto-Park Sign ─────────────────────────────
                    # Currently display-only. Future: trigger parking routine.
                    elif "PARKING" in class_name:
                        if area_ratio >= self.BRAKE_AREA_THRESHOLD:
                            color = (255, 0, 255)  # magenta
                            label = "AUTO-PARK INITIATED"
                        else:
                            color = (200, 0, 200)
                            label = f"PARKING (d:{area_ratio:.3f})"

                    # ── Draw bounding box + label on display ─────────────────
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        display, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                    )

        except Exception as exc:
            print(f"[ZenityBrain] YOLO error: {exc}")

        return stop_close, detections


    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE — TRAFFIC LIGHT CLASSIFIER
    # ══════════════════════════════════════════════════════════════════════════

    def _classify_traffic_light(self, frame_bgr, x1, y1, x2, y2):
        """
        Classifies a traffic light bounding box as RED, GREEN, or UNKNOWN.

        Method:
          1. Crop the bounding box from the frame
          2. Divide vertically into thirds (top = red zone, bottom = green zone)
          3. Convert to HSV colour space
          4. Count saturated red pixels in the top third
          5. Count saturated green pixels in the bottom third
          6. Whichever has more → that's the light state

        HSV ranges:
          Red:   hue 0-10 and 160-180 (red wraps around in HSV)
          Green: hue 40-90

        Returns: (state_string, bgr_color_tuple)
        """
        # Guard against boxes going outside the frame
        fh, fw = frame_bgr.shape[:2]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, fw), min(y2, fh)

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return "UNKNOWN", (200, 200, 200)

        roi_h = roi.shape[0]
        third = max(roi_h // 3, 1)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red spans two hue ranges in OpenCV's HSV (0-180 scale)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 120, 70), (10, 255, 255)),
            cv2.inRange(hsv, (160, 120, 70), (180, 255, 255)),
        )
        green_mask = cv2.inRange(hsv, (40, 70, 70), (90, 255, 255))

        # Count pixels in their respective zones
        red_px = cv2.countNonZero(red_mask[:third, :])       # top third
        green_px = cv2.countNonZero(green_mask[2 * third:, :])  # bottom third

        if red_px > green_px and red_px > 5:
            return "RED", (0, 0, 255)
        elif green_px > red_px and green_px > 5:
            return "GREEN", (0, 255, 0)
        return "UNKNOWN", (200, 200, 200)


    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE — LANE DETECTION  (v8.0 — Nearest-Pair + Lane Memory)
    # ══════════════════════════════════════════════════════════════════════════
    #
    # REWRITE from v7.1's LEFT/RIGHT split.
    #
    # WHY v7.1 FAILED on turns:
    #   On a two-lane track (3 lines: left_solid, centre_dashed, right_solid),
    #   the LEFT/RIGHT split at the image centre worked on straights but broke
    #   on turns. When the inner lane line left the frame, the split picked up
    #   the OUTER road boundary instead, causing the car to track the wrong
    #   lane and drive off the road.
    #
    # NEW APPROACH (v8.0):
    #   1. Find ALL white line positions in the full mask (no left/right split)
    #   2. First frame: pick the pair whose midpoint is closest to the car
    #      centre → this IS the car's lane ("nearest-pair" selection)
    #   3. Subsequent frames: match detected lines to REMEMBERED positions
    #      (lane memory). If a line is too far from its remembered pos,
    #      it's a different line → ignore it. This prevents lane jumping.
    #   4. Camera offset applied so the car centre reference is correct.
    # ══════════════════════════════════════════════════════════════════════════

    def _build_white_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask where white lane lines are 255 and everything
        else is 0. Tuned for BLACK ROAD + WHITE LANE LINES.
        """
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Fixed threshold
        _, mask = cv2.threshold(blur, self.WHITE_THRESH, 255, cv2.THRESH_BINARY)

        # Morphological close — bridge small gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Vertical dilation — bridge gaps between dashes in centre lane divider
        kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        mask = cv2.dilate(mask, kernel_vert, iterations=1)

        return mask

    def _find_all_line_positions(self, mask: np.ndarray):
        """
        Finds the x-centroid of every significant white contour in the mask.

        Returns a sorted list of x-positions (left to right).
        Only returns lines above MIN_CONTOUR_AREA.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        positions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_CONTOUR_AREA:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            positions.append(cx)

        positions.sort()
        return positions

    def _pick_initial_pair(self, line_xs: list, cx_img: int):
        """
        First frame (no lane memory): pick the pair of lines whose midpoint
        is closest to the car centre. On a 3-line track, this naturally picks
        the lane the car is currently in.

        Returns (left_x, right_x) or partial/None.
        """
        if len(line_xs) == 0:
            return None, None

        if len(line_xs) == 1:
            if line_xs[0] < cx_img:
                return line_xs[0], None
            else:
                return None, line_xs[0]

        # Find all pairs with valid lane width
        best = None
        best_score = float('inf')

        for i in range(len(line_xs)):
            for j in range(i + 1, len(line_xs)):
                width = line_xs[j] - line_xs[i]
                if not (self.LANE_WIDTH_MIN < width < self.LANE_WIDTH_MAX):
                    continue
                mid = (line_xs[i] + line_xs[j]) / 2
                score = abs(mid - cx_img)
                if score < best_score:
                    best_score = score
                    best = (line_xs[i], line_xs[j])

        if best is not None:
            return best

        # No valid-width pair — use two nearest to car centre
        sorted_xs = sorted(line_xs, key=lambda x: abs(x - cx_img))
        left = min(sorted_xs[0], sorted_xs[1])
        right = max(sorted_xs[0], sorted_xs[1])
        return left, right

    def _match_to_memory(self, line_xs: list, cx_img: int):
        """
        Match detected lines to remembered lane positions.

        For each remembered position (left/right), find the closest detected
        line within LINE_JUMP_THRESH. If a line is too far from where we
        last saw it, it's a different line (e.g., the outer road boundary)
        and is ignored.

        Falls back to _pick_initial_pair if no memory exists.
        """
        # No memory yet — use initial pair selection
        if self._last_left_x is None and self._last_right_x is None:
            return self._pick_initial_pair(line_xs, cx_img)

        left_x = None
        right_x = None
        used = set()  # indices already assigned

        # Match to remembered LEFT position
        if self._last_left_x is not None:
            best_dist = self.LINE_JUMP_THRESH
            best_idx = -1
            for idx, lx in enumerate(line_xs):
                dist = abs(lx - self._last_left_x)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx >= 0:
                left_x = line_xs[best_idx]
                used.add(best_idx)

        # Match to remembered RIGHT position (skip already-used)
        if self._last_right_x is not None:
            best_dist = self.LINE_JUMP_THRESH
            best_idx = -1
            for idx, lx in enumerate(line_xs):
                if idx in used:
                    continue
                dist = abs(lx - self._last_right_x)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx >= 0:
                right_x = line_xs[best_idx]

        return left_x, right_x

    def _update_lane_memory(self, left_x, right_x):
        """
        Update the EMA-smoothed lane memory with new detections.
        """
        alpha = 0.7  # 70% new value, 30% old

        if left_x is not None:
            if self._last_left_x is None:
                self._last_left_x = left_x
            else:
                self._last_left_x = int(alpha * left_x + (1 - alpha) * self._last_left_x)

        if right_x is not None:
            if self._last_right_x is None:
                self._last_right_x = right_x
            else:
                self._last_right_x = int(alpha * right_x + (1 - alpha) * self._last_right_x)

    def _run_lane_stage(self, frame_bgr, display, h: int, w: int):
        """
        Nearest-pair lane detection with camera offset and lane memory.

        Returns:
            steering_angle (float|None) — PID output in degrees, or None if
                                          both lanes are lost.
        """
        try:
            # 0. Startup skip
            self._frame_count += 1
            if self._frame_count <= self._startup_skip_frames:
                return None

            # 1. Crop the ROI
            roi_y = int(h * self.ROI_TOP_FRACTION)
            roi = frame_bgr[roi_y:h, :]
            roi_h, roi_w = roi.shape[:2]

            # 2. Camera offset: shift "straight ahead" reference
            #    Camera is LEFT of car centre → car centre is RIGHT in image
            cx_img = roi_w // 2 + self.CAMERA_OFFSET_PX
            cx_img = max(0, min(roi_w - 1, cx_img))  # safety clamp

            # 3. Build the white mask
            mask = self._build_white_mask(roi)

            # 4. Draw ROI line and camera offset marker
            cv2.line(display, (0, roi_y), (w, roi_y), (100, 0, 200), 1)
            # Show where the car centre is (after offset) — cyan tick mark
            vis_y = roi_y + roi_h // 2
            cv2.line(display, (cx_img, vis_y - 12), (cx_img, vis_y + 12), (255, 255, 0), 2)

            # 5. Find ALL white line positions in the full mask
            line_xs = self._find_all_line_positions(mask)

            if not line_xs:
                self.pid.reset()
                cv2.putText(
                    display, "NO LANES DETECTED",
                    (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3
                )
                return None

            # 6. Match to lane memory (or pick initial pair on first frame)
            left_x, right_x = self._match_to_memory(line_xs, cx_img)

            # 7. Determine lane centre (target_x)
            half_lane = self._lane_width // 2

            if left_x is not None and right_x is not None:
                # ── BOTH lines visible → best case ───────────────────────────
                target_x = (left_x + right_x) // 2

                # Auto-calibrate lane width
                measured_w = right_x - left_x
                if self.LANE_WIDTH_MIN < measured_w < self.LANE_WIDTH_MAX:
                    self._lane_width = int(0.85 * self._lane_width + 0.15 * measured_w)

                # Visual markers
                cv2.circle(display, (left_x, vis_y), 6, (255, 0, 0), -1)       # blue = left
                cv2.circle(display, (right_x, vis_y), 6, (255, 0, 255), -1)    # magenta = right
                cv2.circle(display, (target_x, vis_y), 6, (0, 255, 255), -1)   # yellow = centre
                cv2.line(display, (left_x, vis_y), (right_x, vis_y), (0, 255, 0), 1)

            elif left_x is not None:
                # ── Only LEFT line visible ────────────────────────────────────
                target_x = left_x + half_lane
                cv2.circle(display, (left_x, vis_y), 6, (0, 165, 255), -1)
                cv2.circle(display, (target_x, vis_y), 4, (0, 200, 200), -1)
                cv2.putText(display, "LEFT ONLY",
                    (10, roi_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            elif right_x is not None:
                # ── Only RIGHT line visible ───────────────────────────────────
                target_x = right_x - half_lane
                cv2.circle(display, (right_x, vis_y), 6, (0, 165, 255), -1)
                cv2.circle(display, (target_x, vis_y), 4, (0, 200, 200), -1)
                cv2.putText(display, "RIGHT ONLY",
                    (10, roi_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            else:
                # ── Both lines lost (shouldn't happen if line_xs is non-empty)
                self.pid.reset()
                return None

            # 8. Update lane memory
            self._update_lane_memory(left_x, right_x)

            # 9. Compute steering error (relative to car centre, not image centre)
            error = target_x - cx_img

            # 10. Deadband
            if abs(error) < self.STEERING_DEADBAND_PX:
                error = 0

            # 11. PID → steering angle
            steering_angle = self.pid.update(error)

            # 12. EMA Smoothing
            if self._frame_count <= self._startup_skip_frames + 1:
                self._steer_ema = steering_angle
            else:
                self._steer_ema = (
                    self._steer_ema_alpha * steering_angle
                    + (1.0 - self._steer_ema_alpha) * self._steer_ema
                )
            steering_angle = self._steer_ema

            # ── HUD ───────────────────────────────────────────────────────────
            cv2.putText(
                display, f"LW:{self._lane_width}px  err:{error:+d}px  lines:{len(line_xs)}",
                (w - 280, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 0), 1
            )
            # Camera offset indicator
            if self.CAMERA_OFFSET_PX != 0:
                cv2.putText(
                    display, f"CAM:{self.CAMERA_OFFSET_PX:+d}px",
                    (w - 280, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1
                )
            # Error arrow
            if target_x != cx_img:
                cv2.arrowedLine(
                    display, (cx_img, vis_y), (target_x, vis_y),
                    (0, 255, 0), 2, tipLength=0.3
                )

            return steering_angle

        except Exception as exc:
            print(f"[ZenityBrain] Lane detection error: {exc}")
            self.pid.reset()
            return None
