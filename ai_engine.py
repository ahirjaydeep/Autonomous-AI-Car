
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
 ZENITY ROV — ai_engine7.py  |  "The Brain"
 Version: 7.0 (Production — Turn-Aware, Single-Lane Tracking)
================================================================================
 KEY IMPROVEMENTS over v4/v5/v6:

 TURN DETECTION (the main problem):
   - Multi-slice scanning (8 slices) with exponential weighting
   - Far-ahead slices weighted HIGHER to see curves EARLY before the car
     reaches them (look-ahead steering)
   - Curve anticipation: when only one lane line is visible (common mid-turn),
     the steering aggressively follows the visible line + inferred offset
   - ROI starts at 35% of frame height (sees further ahead than v4's 50%)
   - Single-lane tracking: car stays in ONE lane on a two-lane track

 YOLO DETECTION:
   - Strict class filtering (Person, Stop, Traffic Light, 30, 40, Parking)
   - Area-ratio proximity triggers (≥4% screen area)
   - Traffic light HSV classifier (red/green pixel thresholding)
   - Stop-sign debounce: 3 consecutive frames required
   - NEW: Returns detection metadata dict (speed_limit, etc.)

 LANE DETECTION:
   - CLAHE + Otsu adaptive thresholding (auto-adjusts to lighting)
   - Morphological cleanup (close gaps in lane lines)
   - Auto-calibrating lane width from real measurements
   - Three-line detection for two-lane tracks (left, center, right)
   - Picks the two nearest lines to form the current lane
   - Weighted centroid steering from multiple horizontal slices

 ARCHITECTURE:
   - All tuneable constants at class level with documentation
   - Separated _build_white_mask() for testability
   - Rich developer logs with [ZenityBrain] prefix
   - Every method documented with purpose + I/O explanation
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
    # TUNEABLE CONSTANTS  (v7.1 — Simplified for Black Road + White Lanes)
    # ══════════════════════════════════════════════════════════════════════════
    # ADJUST THESE FIRST before changing any logic. Each constant has a comment
    # explaining what it does and what happens if you change it.

    # --- WHITE LINE DETECTION ---
    # FIXED threshold — much more reliable than Otsu for white-on-black tracks.
    # Otsu fails here because white lane lines are <5% of the image; the
    # histogram is dominated by black road, so Otsu picks a low threshold
    # that lets road noise through as false lane detections.
    WHITE_THRESH = 180       # ↑ higher (200) = only catches bright white tape/paint
                              # ↓ lower (150) = catches dimmer lines but more noise
                              # For indoor white-tape tracks: 160–190
                              # For outdoor painted lanes: 130–160

    # --- ROI (Region of Interest) ---
    # Controls how much of the frame the lane detector sees.
    # CHANGED from 0.35 → 0.45. Looking too far ahead (0.35) caused the car
    # to steer toward distant features/reflections instead of the actual
    # nearest lane lines. 0.45 = bottom 55% of frame = focuses on the road
    # right in front of the car.
    ROI_TOP_FRACTION = 0.45  # 0.45 = see 55% of frame (near-road focus)
                              # ↓ 0.35 = see further ahead (better for fast curves)
                              # ↑ 0.55 = only immediate road (safest, least anticipation)

    # --- LANE GEOMETRY ---
    LANE_WIDTH_INIT = 260    # initial guess for lane width in pixels
    LANE_WIDTH_MIN  = 80     # reject impossibly narrow line pairs (noise)
    LANE_WIDTH_MAX  = 600    # reject impossibly wide line pairs

    # --- CONTOUR NOISE GATE ---
    MIN_CONTOUR_AREA = 300   # RAISED from 80 → 300. White lane lines are big
                              # continuous blobs. Small blobs are noise.
                              # ↓ lower (100) = catches thin/faint lines
                              # ↑ higher (500) = ultra-clean, may miss narrow markings

    # --- STEERING DEADBAND ---
    # If the lane centre error is within ±DEADBAND pixels of the image centre,
    # output ZERO steering. This prevents jitter and micro-corrections on
    # straight lanes that cause oscillation/U-turns.
    STEERING_DEADBAND_PX = 12  # ±12 pixels = ~2% of a 640px frame
                                # ↑ higher = more stable on straights, slower to react
                                # ↓ lower = more responsive, but may jitter

    # --- YOLO / BRAKING ---
    BRAKE_AREA_THRESHOLD = 0.04   # object occupies ≥4% of frame → trigger action
    STOP_DEBOUNCE_FRAMES = 3      # need 3 consecutive positive frames before braking


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

        # ── EMA steering smoother ─────────────────────────────────────────────
        # Exponential Moving Average filter on the final steering output.
        # Smooths frame-to-frame jitter so the car doesn't oscillate on
        # straight lanes. Alpha = 0.6 means 60% new value, 40% old value.
        self._steer_ema = 0.0        # last smoothed steering angle
        self._steer_ema_alpha = 0.6  # 0.0 = no change, 1.0 = no smoothing

        # ── Startup frame skip ────────────────────────────────────────────────
        # Skip the first N frames of lane detection to let the camera
        # auto-expose and the threshold stabilise. Returns None (drive
        # straight) during these frames.
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
    # PRIVATE — LANE DETECTION  (v7.1 — Simple Left/Right Split)
    # ══════════════════════════════════════════════════════════════════════════
    #
    # COMPLETE REWRITE from v7.0's multi-slice scanner.
    #
    # WHY v7.0 FAILED on straight lanes:
    #   - Otsu threshold was too low for black road + white lanes (white lines
    #     are <5% of the image → Otsu picks a threshold influenced by road noise)
    #   - Multi-slice scanning produced independent noisy detections in 8 slices
    #     with no spatial continuity → random blobs steered the car off course
    #   - PREFERRED_LANE logic picked wrong blob pairs when noise created
    #     extra detections
    #
    # NEW APPROACH (v7.1):
    #   1. Crop the bottom 55% of the frame (ROI)
    #   2. Build a binary white mask using FIXED high threshold (no Otsu)
    #   3. Split the ROI into LEFT HALF and RIGHT HALF at the centre
    #   4. Find the LARGEST contour in each half → that's the lane line
    #   5. Lane centre = midpoint of the two detected line positions
    #   6. Steering error = lane centre − image centre
    #   7. DEADBAND: if error < ±12px → drive perfectly straight (no jitter)
    #   8. Feed error into PID → smooth steering output
    #
    # WHY THIS WORKS:
    #   - Left/right split GUARANTEES left line is found in left half,
    #     right line in right half → zero ambiguity, no wrong picks
    #   - Fixed high threshold (180) ignores all road noise, catches only
    #     bright white lane lines → clean detections
    #   - Deadband eliminates micro-corrections on straight lanes
    #   - Single largest-contour per side ignores small noise blobs
    #
    # TRADE-OFF:
    #   On very tight curves where both lines end up on the same side of
    #   the image, one half loses its line. This is handled by the single-line
    #   fallback (infer the missing line from calibrated lane width).
    # ══════════════════════════════════════════════════════════════════════════

    def _build_white_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask where white lane lines are 255 and everything
        else is 0. Tuned for BLACK ROAD + WHITE LANE LINES.

        Pipeline:
          1. Convert to grayscale
          2. Apply CLAHE (equalises contrast across shadows and bright spots)
          3. Gaussian blur (removes camera sensor noise)
          4. FIXED threshold at WHITE_THRESH (no Otsu — Otsu fails on
             images where white is <5% of the area)
          5. Morphological close with 5×5 kernel (bridges small gaps)
          6. Vertical dilation with tall kernel (bridges gaps between
             dashes in the centre lane divider)

        Returns: binary uint8 mask (0 or 255)
        """
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Fixed threshold — reliable for high-contrast white-on-black tracks
        _, mask = cv2.threshold(blur, self.WHITE_THRESH, 255, cv2.THRESH_BINARY)

        # Step 1: Morphological close — bridge small gaps within each dash/line.
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Step 2: Vertical dilation — bridge gaps BETWEEN dashes.
        # The centre lane divider is dashed (short white segments with gaps).
        # A tall, narrow kernel (3 wide × 15 tall) stretches each dash
        # vertically so nearby dashes merge into one continuous contour.
        # This ensures _find_biggest_contour_x() sees the dashed line as
        # one big blob instead of many small ones.
        kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        mask = cv2.dilate(mask, kernel_vert, iterations=1)

        return mask

    def _find_biggest_contour_x(self, mask_half: np.ndarray, x_offset: int = 0):
        """
        Finds the x-centroid of the LARGEST white contour in a mask region.

        By taking only the largest contour, we reject all small noise blobs
        and guarantee we're tracking the actual lane line (which is always
        the biggest white feature in its half of the image).

        Args:
            mask_half: binary mask of one half (left or right) of the ROI
            x_offset: added to the returned x position (e.g., roi_w//2 for right half)

        Returns:
            int x-position of the largest contour's centroid, or None if
            no valid contour found.
        """
        contours, _ = cv2.findContours(
            mask_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Pick the contour with the largest area
        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)

        if area < self.MIN_CONTOUR_AREA:
            return None  # biggest is still too small → noise frame

        M = cv2.moments(biggest)
        if M["m00"] == 0:
            return None  # degenerate contour

        cx = int(M["m10"] / M["m00"]) + x_offset
        return cx

    def _run_lane_stage(self, frame_bgr, display, h: int, w: int):
        """
        Simple LEFT/RIGHT lane detection with deadband steering.

        Splits the ROI at the centre, finds the biggest white contour in
        each half, and steers to the midpoint.

        Returns:
            steering_angle (float|None) — PID output in degrees, or None if
                                          both lanes are lost.
        """
        try:
            # 0. Startup skip: let the camera auto-expose before engaging PID
            self._frame_count += 1
            if self._frame_count <= self._startup_skip_frames:
                return None  # drive straight during startup

            # 1. Crop the ROI — the bottom portion of the frame
            roi_y = int(h * self.ROI_TOP_FRACTION)
            roi = frame_bgr[roi_y:h, :]
            roi_h, roi_w = roi.shape[:2]
            cx_img = roi_w // 2  # image centre ("straight ahead" reference)

            # 2. Build the white mask
            mask = self._build_white_mask(roi)

            # 3. Draw the ROI horizon line on the display
            cv2.line(display, (0, roi_y), (w, roi_y), (100, 0, 200), 1)

            # 4. Split into LEFT and RIGHT halves at the centre
            left_mask  = mask[:, :cx_img]
            right_mask = mask[:, cx_img:]

            # 5. Find the biggest contour in each half
            left_x  = self._find_biggest_contour_x(left_mask, x_offset=0)
            right_x = self._find_biggest_contour_x(right_mask, x_offset=cx_img)

            # 6. Determine lane centre (target_x)
            half_lane = self._lane_width // 2
            vis_y = roi_y + roi_h // 2  # y-position for visual markers

            if left_x is not None and right_x is not None:
                # ── BOTH lines visible → best case ───────────────────────────
                target_x = (left_x + right_x) // 2

                # Auto-calibrate lane width from real measurement
                measured_w = right_x - left_x
                if self.LANE_WIDTH_MIN < measured_w < self.LANE_WIDTH_MAX:
                    self._lane_width = int(0.85 * self._lane_width + 0.15 * measured_w)

                # Visual markers on dashboard
                cv2.circle(display, (left_x, vis_y), 6, (255, 0, 0), -1)       # blue = left line
                cv2.circle(display, (right_x, vis_y), 6, (255, 0, 255), -1)    # magenta = right line
                cv2.circle(display, (target_x, vis_y), 6, (0, 255, 255), -1)   # yellow = lane centre
                cv2.line(display, (left_x, vis_y), (right_x, vis_y), (0, 255, 0), 1)

            elif left_x is not None:
                # ── Only LEFT line visible (right lost in turn) ──────────────
                target_x = left_x + half_lane
                cv2.circle(display, (left_x, vis_y), 6, (0, 165, 255), -1)   # orange
                cv2.circle(display, (target_x, vis_y), 4, (0, 200, 200), -1)
                cv2.putText(
                    display, "LEFT ONLY",
                    (10, roi_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1
                )

            elif right_x is not None:
                # ── Only RIGHT line visible (left lost in turn) ──────────────
                target_x = right_x - half_lane
                cv2.circle(display, (right_x, vis_y), 6, (0, 165, 255), -1)  # orange
                cv2.circle(display, (target_x, vis_y), 4, (0, 200, 200), -1)
                cv2.putText(
                    display, "RIGHT ONLY",
                    (10, roi_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1
                )

            else:
                # ── NO lines found → total lane loss ─────────────────────────
                self.pid.reset()
                cv2.putText(
                    display, "NO LANES DETECTED",
                    (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3
                )
                return None

            # 7. Compute steering error
            error = target_x - cx_img

            # 8. DEADBAND: if error is tiny, drive perfectly straight.
            #    This prevents jitter/oscillation on straight lanes.
            if abs(error) < self.STEERING_DEADBAND_PX:
                error = 0

            # 9. PID → steering angle
            steering_angle = self.pid.update(error)

            # 10. EMA Smoothing — blend with previous to reduce jitter
            if self._frame_count <= self._startup_skip_frames + 1:
                self._steer_ema = steering_angle  # seed first valid frame
            else:
                self._steer_ema = (
                    self._steer_ema_alpha * steering_angle
                    + (1.0 - self._steer_ema_alpha) * self._steer_ema
                )
            steering_angle = self._steer_ema

            # ── HUD debug info ────────────────────────────────────────────────
            cv2.putText(
                display, f"LW:{self._lane_width}px  err:{error:+d}px",
                (w - 220, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 0), 1
            )

            # Draw error line from centre to target
            cv2.line(display, (cx_img, vis_y - 8), (cx_img, vis_y + 8), (200, 200, 200), 2)
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
