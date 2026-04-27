"""
================================================================================
 ZENITY ROV — main_rov7.py  |  "The Nervous System"
 Version: 7.0 (Production-Grade — Full State Machine)
================================================================================
 Responsibilities:
   - Camera thread:    pulls /shot.jpg from IP Webcam in a tight loop
   - Heartbeat thread: sends PING to ESP32 at 2 Hz (watchdog keep-alive)
   - UDP socket:       fires "L,R\n" PWM to ESP32 at up to 10 Hz
   - State machine:    DRIVE → STOP → COOLDOWN → DRIVE (with LOST recovery)
   - Speed zones:      dynamically adjusts BASE_SPEED from 30/40 sign detections
   - HUD overlay:      real-time dashboard with state, steering, perf stats
   - Graceful shutdown: sends STOP to ESP32 before Python exits

 UDP Packet format (UTF-8, newline-terminated):
   Normal drive  →  "180,150\n"    (left_pwm, right_pwm)
   Full stop     →  "0,0\n"
   Reverse       →  "-100,-100\n"  (negative = backward)
   Heartbeat     →  "PING\n"

 Compatible with: esp_rov7.ino (ESP32 firmware)
 AI Engine:       ai_engine7.py  (ZenityBrain v7.0)
================================================================================
"""

import cv2
import time
import queue
import socket
import threading
import requests
import numpy as np
import signal
import sys
from collections import deque

try:
    import kornia_rs as K
    import torch
    _USE_KORNIA = True
except ImportError:
    _USE_KORNIA = False
    print("[main] kornia_rs not found — using cv2.imdecode (slower)")

from ai_engine7 import ZenityBrain


# ══════════════════════════════════════════════════════════════════════════════
# ▌ CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
# ONLY this section should change between runs. Edit the IPs to match your
# current network, adjust BASE_SPEED to taste, and you're ready to go.

PHONE_IP   = "192.168.31.129"   # IP Webcam phone (Android app)
ESP32_IP   = "10.48.167.62"    # ESP32 address (shown in Serial Monitor)
ESP32_PORT = 4210              # UDP port — MUST match esp_rov7.ino

STREAM_URL = f"http://{PHONE_IP}:8080/shot.jpg"

# ── Speed constants (0–255 PWM) ───────────────────────────────────────────────
# BASE_SPEED controls how fast the car drives straight. This is the
# default; it can be overridden by speed zone detections (30/40 signs).
BASE_SPEED = 90      # Default forward speed (used until a speed sign is seen)
SLOW_SPEED = 55      # Speed during lane-lost recovery crawl

# ── Speed Zone Mapping ───────────────────────────────────────────────────────
# When a speed sign is detected close enough, BASE_SPEED is temporarily
# changed to this value. The car returns to DEFAULT_BASE_SPEED if no zone
# is detected for SPEED_ZONE_TIMEOUT seconds.
SPEED_ZONE_MAP = {
    30: 70,    # 30 sign → slow to PWM 70 (roughly 30% of max)
    40: 90,    # 40 sign → PWM 90 (default speed — no change needed at 90)
}
DEFAULT_BASE_SPEED    = 90     # Speed when no zone is active
SPEED_ZONE_TIMEOUT    = 8.0   # Seconds after last sign detection to reset speed

# ── Startup grace period ─────────────────────────────────────────────────────
# Number of frames to drive straight (BASE_SPEED, BASE_SPEED) before engaging
# PID steering. This lets the lane detection stabilise and prevents the
# derivative term from spiking on the very first frame.
STARTUP_GRACE_FRAMES  = 5

# ── Behaviour timings ─────────────────────────────────────────────────────────
MAX_STOP_DURATION     = 3.0   # Seconds to hold brake after stop sign detected
COOLDOWN_DURATION     = 3.0   # Seconds to ignore stop signs after resuming
                               # (prevents re-triggering on the same sign)
LOST_CRAWL_TIMEOUT    = 2.0   # Seconds of lane loss before entering LOST state
COMMAND_HZ            = 10    # Motor command rate (Hz) — 100ms between commands
HEARTBEAT_HZ          = 2     # ESP32 keep-alive rate (Hz)
CAMERA_TIMEOUT_S      = 2.0   # Seconds with no frame → camera-down warning

# ── Steering sensitivity ──────────────────────────────────────────────────────
# The PID output (degrees) is divided by this to get a normalised [-1, 1] value.
# Lower = sharper turns. Higher = gentler steering.
STEER_DIVISOR = 35.0

# ── Minimum wheel speed during turns ──────────────────────────────────────────
# The slower wheel never drops below this PWM value. This prevents the car
# from pivoting on one wheel (which looks like a U-turn). Instead, the slow
# wheel keeps spinning forward slowly, making the car ARC through the turn.
#   Too LOW (0-20)  → car can still pivot on tight turns
#   Too HIGH (60+)  → car can't turn sharply enough for tight curves
MIN_TURN_PWM = 40


# ══════════════════════════════════════════════════════════════════════════════
# ▌ DRIVE STATE MACHINE
# ══════════════════════════════════════════════════════════════════════════════
#
# State diagram:
#
#   ┌─────────┐  stop detected  ┌─────────┐  3s elapsed  ┌──────────┐  3s elapsed  ┌─────────┐
#   │  DRIVE  │ ───────────────> │  STOP   │ ────────────>│ COOLDOWN │ ────────────> │  DRIVE  │
#   └─────────┘                  └─────────┘              └──────────┘              └─────────┘
#       │                                                      │
#       │  lane lost > 2s                                      │ stop signs IGNORED
#       ▼                                                      │ persons/red lights STILL TRIGGER
#   ┌─────────┐                                                │
#   │  LOST   │ ── lane reacquired ───────────────────────────>│
#   └─────────┘                                                │
#       │                                                      │
#       │  stop detected (safety)                              │
#       ▼                                                      │
#   ┌─────────┐                                                │
#   │  STOP   │ ──────────────────────────────────────────────>│
#   └─────────┘
#
# ══════════════════════════════════════════════════════════════════════════════

class DriveState:
    DRIVE    = "DRIVE"      # Normal lane-following
    STOP     = "STOP"       # Braking (stop sign / red light / person)
    COOLDOWN = "COOLDOWN"   # Post-stop immunity (ignores stop signs only)
    LOST     = "LOST"       # Lane totally gone — crawl and search


class StateMachine:
    """
    Converts raw AI outputs into clean, debounced motor commands.
    Handles all state transitions including stop-sign cooldown.

    The cooldown feature prevents the car from stopping twice at the same
    stop sign: after the 3-second stop, the car drives for another 3 seconds
    while ignoring stop sign detections. Person and red light detections
    are NEVER suppressed — safety overrides convenience.
    """

    def __init__(self):
        self.state = DriveState.DRIVE
        self._stop_ts = None       # timestamp when STOP was entered
        self._cooldown_ts = None   # timestamp when COOLDOWN was entered
        self._lost_ts = None       # timestamp when lane was first lost

        # Dynamic speed: can be changed by speed zone detections
        self.active_base_speed = DEFAULT_BASE_SPEED
        self._last_speed_zone_ts = 0.0  # when the last speed zone sign was seen

        # Startup grace: drive straight for the first N frames
        self._startup_frames_remaining = STARTUP_GRACE_FRAMES

    def update(self, stop_detected: bool, steering, detections: dict):
        """
        Process one frame's AI results and return motor commands.

        Args:
            stop_detected: True if debounced stop condition confirmed by AI
            steering: PID steering angle (float, deg) or None if lanes lost
            detections: dict with keys 'speed_limit', 'person_close',
                        'stop_sign_close', 'red_light'

        Returns:
            (left_pwm, right_pwm) — motor speeds (may be negative for reverse)
        """
        now = time.time()

        # ── Speed Zone Handling (independent of state) 
        # If a speed sign was detected close enough, adjust driving speed.
        # This is checked in every state because we want the speed to update
        # even during COOLDOWN or after LOST recovery.
        self._handle_speed_zone(detections, now)

        # ── DRIVE state 
        if self.state == DriveState.DRIVE:
            # Check for stop-triggering conditions
            if stop_detected:
                self._enter_stop(now)
                return 0, 0

            # Check for lane loss
            if steering is None:
                if self._lost_ts is None:
                    self._lost_ts = now
                    print("[StateMachine] Lane lost — starting grace period")

                elapsed_lost = now - self._lost_ts
                if elapsed_lost > LOST_CRAWL_TIMEOUT:
                    print(f"[StateMachine] {elapsed_lost:.1f}s lane loss → LOST state")
                    self.state = DriveState.LOST
                    return 0, 0  # full stop — too dangerous to crawl blind

                # Within grace period: crawl straight slowly
                return SLOW_SPEED, SLOW_SPEED

            # Startup grace: drive straight for the first N frames to let PID settle
            if self._startup_frames_remaining > 0:
                self._startup_frames_remaining -= 1
                bs = self.active_base_speed
                return bs, bs

            # Normal driving: lane acquired, no stop condition
            self._lost_ts = None
            left, right = self._steering_to_tank(steering)
            return left, right

        # ── STOP state 
        elif self.state == DriveState.STOP:
            elapsed = now - self._stop_ts
            if elapsed >= MAX_STOP_DURATION:
                print(f"[StateMachine] Stop held {elapsed:.1f}s → COOLDOWN (ignoring stop signs for {COOLDOWN_DURATION}s)")
                self.state = DriveState.COOLDOWN
                self._cooldown_ts = now
                self._lost_ts = None
            return 0, 0  # motors stay off during entire STOP phase

        # ── COOLDOWN state 
        # Car is driving but ignoring stop signs. Still reacts to persons
        # and red lights (safety is never suppressed).
        elif self.state == DriveState.COOLDOWN:
            elapsed = now - self._cooldown_ts

            # Check if cooldown period is over
            if elapsed >= COOLDOWN_DURATION:
                print(f"[StateMachine] Cooldown expired → DRIVE (all detections active)")
                self.state = DriveState.DRIVE
                self._lost_ts = None
                # Fall through to normal driving below

            # During cooldown: only person and red light can stop us
            if detections.get('person_close') or detections.get('red_light'):
                print("[StateMachine] ⚠ Safety override during COOLDOWN — person/red light!")
                self._enter_stop(now)
                return 0, 0

            # Normal driving (stop signs are suppressed)
            if steering is None:
                if self._lost_ts is None:
                    self._lost_ts = now
                elapsed_lost = now - self._lost_ts
                if elapsed_lost > LOST_CRAWL_TIMEOUT:
                    self.state = DriveState.LOST
                    return 0, 0  # full stop
                return SLOW_SPEED, SLOW_SPEED

            self._lost_ts = None
            left, right = self._steering_to_tank(steering)
            return left, right

        # ── LOST state 
        elif self.state == DriveState.LOST:
            # Safety: even while lost, react to stop conditions
            if stop_detected:
                self._enter_stop(now)
                return 0, 0

            # If lanes reappear, resume driving immediately
            if steering is not None:
                print("[StateMachine] Lane reacquired → DRIVE")
                self.state = DriveState.DRIVE
                self._lost_ts = None
                left, right = self._steering_to_tank(steering)
                return left, right

            # Still lost — full stop (safer than driving blind)
            return 0, 0

        # Fallback safety net (should never reach here)
        return 0, 0

    def _enter_stop(self, ts: float):
        """Transition to STOP state."""
        self.state = DriveState.STOP
        self._stop_ts = ts
        print("[StateMachine] → STOP (braking)")

    def _handle_speed_zone(self, detections: dict, now: float):
        """
        Dynamically adjusts base speed based on detected speed limit signs.
        Resets to default if no sign has been seen for SPEED_ZONE_TIMEOUT seconds.
        """
        speed_limit = detections.get('speed_limit')

        if speed_limit is not None and speed_limit in SPEED_ZONE_MAP:
            new_speed = SPEED_ZONE_MAP[speed_limit]
            if self.active_base_speed != new_speed:
                print(f"[StateMachine] Speed zone detected: {speed_limit} → BASE_SPEED={new_speed}")
            self.active_base_speed = new_speed
            self._last_speed_zone_ts = now

        elif self._last_speed_zone_ts > 0 and (now - self._last_speed_zone_ts) > SPEED_ZONE_TIMEOUT:
            # No speed sign seen for a while — reset to default
            if self.active_base_speed != DEFAULT_BASE_SPEED:
                print(f"[StateMachine] Speed zone timeout → BASE_SPEED={DEFAULT_BASE_SPEED}")
            self.active_base_speed = DEFAULT_BASE_SPEED

    def _steering_to_tank(self, steering_deg: float):
        """
        Maps a signed PID steering angle to differential (tank / skid-steer) PWM.

        PERCENTAGE-BASED FORMULA (v7.1 fix):
          Instead of adding a fixed TURN_POWER offset (which caused U-turns
          when TURN_POWER > BASE_SPEED), we now scale each wheel as a
          percentage of BASE_SPEED:
            left  = BASE_SPEED × (1 + t)
            right = BASE_SPEED × (1 - t)

          MIN_TURN_PWM floor ensures the slow wheel never stops completely,
          preventing pivot turns that look like U-turns.

          At t=0 (straight):    L=90,  R=90   (perfect straight line)
          At t=0.5 (moderate):  L=135, R=45   (gentle turn)
          At t=1.0 (max steer): L=180, R=40   (sharp arc, NOT a pivot)

        Direction convention:
          steering_deg > 0 → turn RIGHT → left motor faster, right slower
          steering_deg < 0 → turn LEFT  → right motor faster, left slower
        """
        # Normalise to [-1, 1] — STEER_DIVISOR controls sensitivity
        t = max(-1.0, min(1.0, steering_deg / STEER_DIVISOR))

        # Percentage-based differential
        bs = self.active_base_speed
        left  = int(bs * (1.0 + t))
        right = int(bs * (1.0 - t))

        # Clamp: MIN_TURN_PWM floor prevents pivot turns (one wheel at 0),
        # 255 ceiling prevents exceeding PWM range.
        return (max(MIN_TURN_PWM, min(255, left)),
                max(MIN_TURN_PWM, min(255, right)))


# ▌ HELPERS

def _send_udp(sock: socket.socket, addr: tuple, payload: str):
    """
    Fire-and-forget UDP send. Non-blocking, silently absorbs errors.
    For motor control, a dropped packet is better than a blocking one
    (the car just holds its previous heading for one cycle).
    """
    try:
        sock.sendto(payload.encode(), addr)
    except OSError:
        pass


# ▌ CAMERA THREAD

def camera_thread(q: queue.Queue, stop_event: threading.Event):
    """
    Background thread that pulls /shot.jpg from the phone in a tight loop.

    Only the NEWEST frame is kept — old frames are immediately discarded.
    This eliminates buffer bloat: the AI always processes the most current
    view, not a frame from 500ms ago.

    Decoding priority:
      1. kornia_rs (Rust-native, DLPack zero-copy) — ~2x faster on Apple Silicon
      2. cv2.imdecode (C++ OpenCV) — fallback if kornia not installed

    Error handling:
      Exponential backoff on failures (caps at 1 second) to avoid hammering
      the phone when it's temporarily unreachable.
    """
    session = requests.Session()
    session.headers.update({"Connection": "keep-alive"})
    decoder = (K.ImageDecoder() if _USE_KORNIA else None)
    backoff = 0.05
    consecutive_errors = 0

    while not stop_event.is_set():
        try:
            resp = session.get(STREAM_URL, timeout=1.5)
            resp.raise_for_status()

            # ── Decode the JPEG bytes to a BGR numpy array 
            if _USE_KORNIA and decoder is not None:
                decoded = decoder.decode(resp.content)
                img_bgr = cv2.cvtColor(
                    torch.from_dlpack(decoded).numpy(), cv2.COLOR_RGB2BGR
                )
            else:
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                raise ValueError("Decoded frame is None (corrupt JPEG?)")

            # ── Always keep only the newest frame 
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put(img_bgr)

            backoff = 0.05
            consecutive_errors = 0

        except Exception as exc:
            consecutive_errors += 1
            if consecutive_errors <= 3 or consecutive_errors % 20 == 0:
                print(f"[Camera] Error #{consecutive_errors}: {exc} — retry in {backoff:.2f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 1.0)


# ▌ HEARTBEAT THREAD

def heartbeat_thread(sock: socket.socket, addr: tuple, stop_event: threading.Event):
    """
    Sends "PING\n" to the ESP32 at HEARTBEAT_HZ (default: 2 Hz).

    Purpose: proves to the ESP32 that the Python brain is alive. If this
    thread stops (Python crash, freeze, Wi-Fi drop), the ESP32's watchdog
    timer fires and kills the motors automatically.

    The heartbeat is sent on the SAME UDP socket as motor commands.
    """
    interval = 1.0 / HEARTBEAT_HZ
    while not stop_event.is_set():
        _send_udp(sock, addr, "PING\n")
        time.sleep(interval)


# ▌ PERFORMANCE TRACKER

class PerfTracker:
    """
    Tracks rolling-window FPS and average AI latency for the HUD.

    Uses a deque of timestamps to compute frames-per-second over the last
    N frames (default: 30). This smooths out individual frame spikes
    and gives a stable readout.
    """

    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)
        self._latencies: deque = deque(maxlen=window)

    def tick(self, latency_ms: float):
        """Record one frame's timestamp and AI latency."""
        self._times.append(time.perf_counter())
        self._latencies.append(latency_ms)

    @property
    def fps(self) -> float:
        """Smoothed frames-per-second over the rolling window."""
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0] + 1e-9)

    @property
    def avg_latency(self) -> float:
        """Average AI processing time in milliseconds."""
        return sum(self._latencies) / max(len(self._latencies), 1)


# ▌ HUD OVERLAY

# Style mapping: state → (label_text, BGR_color)
_STATE_STYLE = {
    DriveState.DRIVE:    ("DRIVE",    (0, 220, 0)),      # green
    DriveState.STOP:     ("STOP",     (0, 0, 255)),      # red
    DriveState.COOLDOWN: ("COOLDOWN", (255, 200, 0)),    # cyan-ish
    DriveState.LOST:     ("LOST",     (0, 165, 255)),    # orange
}


def _draw_hud(display, state_machine, left, right, steering, latency_ms, fps):
    """
    Draws a clean informational overlay on the dashboard frame.

    Elements:
      - Top banner: state, motor PWMs, steering angle, speed zone
      - Bottom-right: FPS and AI latency
      - Bottom-centre: steering gauge bar (visual representation)
    """
    h, w = display.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    state = state_machine.state

    # ── Top banner (semi-transparent black) 
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    # State label
    label, color = _STATE_STYLE.get(state, ("???", (200, 200, 200)))
    cv2.putText(display, f"[{label}]", (8, 34), font, 0.85, color, 2)

    # Motor PWMs
    cv2.putText(
        display, f"L:{left:+4d}  R:{right:+4d}",
        (170, 34), font, 0.68, (255, 255, 255), 2
    )

    # Steering angle
    steer_str = f"Steer:{steering:+6.1f}°" if steering is not None else "Steer:  LOST"
    steer_col = (200, 200, 255) if steering is not None else (0, 100, 255)
    cv2.putText(display, steer_str, (400, 34), font, 0.62, steer_col, 2)

    # Active speed zone
    spd = state_machine.active_base_speed
    if spd != DEFAULT_BASE_SPEED:
        cv2.putText(
            display, f"ZONE:{spd}",
            (w - 130, 34), font, 0.6, (0, 255, 255), 2
        )

    # ── Bottom-right: performance stats 
    cv2.putText(
        display, f"{fps:.1f} FPS  {latency_ms:.1f}ms",
        (w - 220, h - 10), font, 0.6, (100, 255, 100), 1
    )

    # ── Bottom-centre: steering gauge bar 
    # A horizontal bar that shows the steering direction visually.
    bar_cx = w // 2
    bar_y = h - 20
    bar_hw = 80  # half-width of the gauge

    # Background bar (dark grey)
    cv2.rectangle(
        display, (bar_cx - bar_hw, bar_y - 6),
        (bar_cx + bar_hw, bar_y + 6), (50, 50, 50), -1
    )

    # Steering indicator (filled towards the turn direction)
    if steering is not None:
        norm = np.clip(steering / 35.0, -1.0, 1.0)
        indicator_x = int(bar_cx + norm * bar_hw)
        bar_color = (0, 200, 255) if abs(steering) < 15 else (0, 80, 255)
        cv2.rectangle(
            display, (bar_cx, bar_y - 6),
            (indicator_x, bar_y + 6), bar_color, -1
        )

    # Centre tick mark
    cv2.line(display, (bar_cx, bar_y - 10), (bar_cx, bar_y + 10), (200, 200, 200), 1)


# ▌ MAIN LOOP

def main():
    print("=" * 64)
    print("  ZENITY ROV  |  v7.0  |  Initialising…")
    print("=" * 64)
    print(f"  Phone IP:  {PHONE_IP}")
    print(f"  ESP32 IP:  {ESP32_IP}:{ESP32_PORT}")
    print(f"  Base Speed: {BASE_SPEED}   Steer Divisor: {STEER_DIVISOR}")
    print(f"  Stop Duration: {MAX_STOP_DURATION}s   Cooldown: {COOLDOWN_DURATION}s")
    print(f"  kornia_rs: {'✓ enabled' if _USE_KORNIA else '✗ disabled (using cv2)'}")
    print("=" * 64)

    # ── Subsystem init 
    ai_brain = ZenityBrain()               # AI perception engine
    state_machine = StateMachine()          # State machine + speed zones
    perf = PerfTracker()                    # FPS / latency monitor
    stop_event = threading.Event()          # Shared stop signal for all threads

    # ── UDP socket (shared by main loop + heartbeat thread) 
    esp_addr = (ESP32_IP, ESP32_PORT)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setblocking(False)   # never block the main loop waiting for network

    # ── Graceful shutdown handler 
    # Catches Ctrl+C and SIGTERM; sends 3 stop commands for reliability.
    def _shutdown(sig=None, frame=None):
        print("\n[main] Shutdown signal — sending STOP to ESP32 …")
        stop_event.set()
        for _ in range(3):           # triple-send for reliability on lossy UDP
            _send_udp(udp_sock, esp_addr, "0,0\n")
            time.sleep(0.05)
        udp_sock.close()
        cv2.destroyAllWindows()
        print("[main] Clean exit. ✓")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Camera + Heartbeat threads 
    frame_queue = queue.Queue(maxsize=2)

    t_cam = threading.Thread(
        target=camera_thread,
        args=(frame_queue, stop_event),
        daemon=True,
    )
    t_hb = threading.Thread(
        target=heartbeat_thread,
        args=(udp_sock, esp_addr, stop_event),
        daemon=True,
    )
    t_cam.start()
    t_hb.start()
    print("[main] Camera + heartbeat threads started — waiting for first frame …\n")

    # ── Main loop state 
    last_cmd_time = 0.0
    cmd_interval = 1.0 / COMMAND_HZ
    frame_count = 0
    left_pwm = 0
    right_pwm = 0

    while not stop_event.is_set():

        # ── Frame acquisition 
        try:
            frame = frame_queue.get(timeout=CAMERA_TIMEOUT_S)
        except queue.Empty:
            print("[main] ⚠ No frame for {:.1f}s — camera down? Sending stop.".format(CAMERA_TIMEOUT_S))
            _send_udp(udp_sock, esp_addr, "0,0\n")
            continue

        t0 = time.perf_counter()

        # ── AI inference 
        # Returns: annotated display frame, steering angle, stop flag, detections dict
        display, steering, stop_detected, detections = ai_brain.process_frame(frame)
        ai_ms = (time.perf_counter() - t0) * 1000

        if display is None:
            continue

        # ── State machine → motor command 
        left_pwm, right_pwm = state_machine.update(stop_detected, steering, detections)

        # ── Rate-limited UDP send 
        now = time.perf_counter()
        if now - last_cmd_time >= cmd_interval:
            last_cmd_time = now
            _send_udp(udp_sock, esp_addr, f"{left_pwm},{right_pwm}\n")

            # ── Developer console log (every 10th command) 
            frame_count += 1
            if frame_count % 10 == 0:
                state_str = state_machine.state
                steer_str = f"{steering:+.1f}°" if steering is not None else "N/A"
                speed_info = f"SPD:{state_machine.active_base_speed}"
                zone_info = ""
                if detections.get('speed_limit'):
                    zone_info = f"  ZONE:{detections['speed_limit']}"
                if detections.get('person_close'):
                    zone_info += "  ⚠PERSON"
                if detections.get('red_light'):
                    zone_info += "  🔴LIGHT"

                print(
                    f"[{state_str:<8}]  L:{left_pwm:+4d}  R:{right_pwm:+4d}"
                    f"  steer:{steer_str:>8}  {speed_info}"
                    f"  AI:{ai_ms:5.1f}ms  FPS:{perf.fps:4.1f}{zone_info}"
                )

        # ── Dashboard overlay 
        perf.tick(ai_ms)
        _draw_hud(display, state_machine, left_pwm, right_pwm, steering, ai_ms, perf.fps)

        cv2.imshow("Zenity ROV — v7.0", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[main] 'q' pressed — shutting down")
            break
        elif key == ord(" "):
            # Manual emergency stop: press SPACE to immediately halt
            print("[main] ⚠ Manual E-STOP (SPACE key)")
            _send_udp(udp_sock, esp_addr, "0,0\n")

    _shutdown()

if __name__ == "__main__":
    main()
