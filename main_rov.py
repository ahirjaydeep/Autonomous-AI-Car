"""
================================================================================
 ZENITY ROV — main_rov.py  |  "The Nervous System"
 Version: 3.0 (Production-Grade — Refined)
================================================================================
 Responsibilities:
   - Camera thread: aggressive pull from IP Webcam /shot.jpg  (no buffer bloat)
   - UDP command socket: fires L,R PWM strings to ESP32 at ≤10 ms overhead
   - Heartbeat thread: sends keep-alive every 500 ms; ESP32 E-STOPs if silent
   - Drive state machine: DRIVE → STOP → RESUME  (debounced, clean transitions)
   - LOST state: recovers from total lane loss by slowing + searching
   - FPS counter and latency stats on HUD
   - Graceful shutdown: sends STOP before Python exits

 UDP Packet format (UTF-8):
   Normal drive  →  "180,150\n"
   Full stop     →  "0,0\n"
   Heartbeat     →  "PING\n"

 ESP32 C++ watchdog snippet:
   if (millis() - lastPing > 1000) { motor_L(0); motor_R(0); }
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
    print("[main] kornia_rs not found — using cv2.imdecode")

from ai_engine6 import ZenityBrain


# ▌ CONFIGURATION  ← only file you should need to edit between runs

PHONE_IP   = "10.48.167.163"   
ESP32_IP   = "10.48.167.62"    
ESP32_PORT = 4210              

STREAM_URL = f"http://{PHONE_IP}:8080/shot.jpg"

# ── Speed constants (0–255) 
BASE_SPEED          = 90   
SLOW_SPEED          = 55    
TURN_POWER          = 110   

# ── Behaviour timings 
MAX_STOP_DURATION   = 3.0   
LOST_CRAWL_TIMEOUT  = 2.0   
COMMAND_HZ          = 10    
HEARTBEAT_HZ        = 2     
CAMERA_TIMEOUT_S    = 2.0   

# ── Steering sensitivity 
STEER_DIVISOR       = 15.0  

# ▌ DRIVE STATE MACHINE

class DriveState:
    DRIVE = "DRIVE"   
    STOP  = "STOP"    
    LOST  = "LOST"    


class StateMachine:

    def __init__(self):
        self.state       = DriveState.DRIVE
        self._stop_ts    = None
        self._lost_ts    = None

    def update(self, stop_detected: bool, steering):
        now = time.time()

        # ── DRIVE ─────────────────────────────────────────────────────────────
        if self.state == DriveState.DRIVE:
            if stop_detected:
                self._enter_stop(now)
                return 0, 0

            if steering is None:
                # First frame of lane loss — start the lost timer
                if self._lost_ts is None:
                    self._lost_ts = now

                elapsed_lost = now - self._lost_ts
                if elapsed_lost > LOST_CRAWL_TIMEOUT:
                    print(f"[StateMachine] {elapsed_lost:.1f}s lane loss → LOST state")
                    self.state = DriveState.LOST
                    return 0, 0

                # Still within grace period — crawl straight
                return SLOW_SPEED, SLOW_SPEED

            # Normal: lane acquired
            self._lost_ts = None
            left, right   = _steering_to_tank(steering)
            return left, right

        # ── STOP 
        elif self.state == DriveState.STOP:
            elapsed = now - self._stop_ts
            if elapsed >= MAX_STOP_DURATION:
                print(f"[StateMachine] Stop held {elapsed:.1f}s → DRIVE")
                self.state    = DriveState.DRIVE
                self._lost_ts = None
            return 0, 0

        # ── LOST 
        elif self.state == DriveState.LOST:
            if stop_detected:
                self._enter_stop(now)
                return 0, 0
            if steering is not None:
                print("[StateMachine] Lane reacquired → DRIVE")
                self.state    = DriveState.DRIVE
                self._lost_ts = None
                left, right   = _steering_to_tank(steering)
                return left, right
            # Still lost — full stop (safer than crawling blind)
            return 0, 0

        return 0, 0   # unreachable safety net

    def _enter_stop(self, ts: float):
        self.state    = DriveState.STOP
        self._stop_ts = ts
        print("[StateMachine] STOP triggered")


# ▌ HELPERS

def _steering_to_tank(steering_deg: float):
    """
    Maps signed PID steering angle → differential (tank / skid-steer) PWM.

      steering_deg > 0 → turn RIGHT  → right motor slower (or reverse)
      steering_deg < 0 → turn LEFT   → left motor slower (or reverse)

    Allowing negative values means the inner wheel spins backwards on sharp
    turns, giving a true pivot — essential for a 4WD skid-steer chassis.
    """
    t     = max(-1.0, min(1.0, steering_deg / STEER_DIVISOR))
    left  = int(BASE_SPEED + t * TURN_POWER)
    right = int(BASE_SPEED - t * TURN_POWER)
    return (max(-255, min(255, left)),
            max(-255, min(255, right)))


def _send_udp(sock: socket.socket, addr: tuple, payload: str):
    """Non-blocking fire-and-forget UDP. Silently absorbs network errors."""
    try:
        sock.sendto(payload.encode(), addr)
    except OSError:
        pass


# ▌ CAMERA THREAD

def camera_thread(q: queue.Queue, stop_event: threading.Event):
    """
    Pulls /shot.jpg in a tight loop, always dropping stale frames.
    Uses kornia_rs Rust decoder if available, otherwise cv2.imdecode.
    Exponential back-off on failure (caps at 1 s).
    """
    session = requests.Session()
    session.headers.update({"Connection": "keep-alive"})
    decoder = (K.ImageDecoder() if _USE_KORNIA else None)
    backoff = 0.05

    while not stop_event.is_set():
        try:
            resp = session.get(STREAM_URL, timeout=1.5)
            resp.raise_for_status()

            if _USE_KORNIA and decoder is not None:
                decoded = decoder.decode(resp.content)
                img_bgr = cv2.cvtColor(
                    torch.from_dlpack(decoded).numpy(), cv2.COLOR_RGB2BGR)
            else:
                arr     = np.frombuffer(resp.content, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                raise ValueError("Decoded frame is None")

            # Always hold only the newest frame
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put(img_bgr)
            backoff = 0.05

        except Exception as exc:
            print(f"[Camera] {exc}  — retry in {backoff:.2f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 1.0)


# ▌ HEARTBEAT THREAD

def heartbeat_thread(sock: socket.socket, addr: tuple, stop_event: threading.Event):
    """
    Sends "PING\\n" to ESP32 at HEARTBEAT_HZ.
    If Python crashes, this thread dies → ESP32 watchdog fires E-STOP.
    """
    interval = 1.0 / HEARTBEAT_HZ
    while not stop_event.is_set():
        _send_udp(sock, addr, "PING\n")
        time.sleep(interval)


# ▌ FPS / LATENCY TRACKER

class PerfTracker:
    """Rolling window FPS and average AI latency display."""

    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)
        self._latencies: deque = deque(maxlen=window)

    def tick(self, latency_ms: float):
        self._times.append(time.perf_counter())
        self._latencies.append(latency_ms)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0] + 1e-9)

    @property
    def avg_latency(self) -> float:
        return sum(self._latencies) / max(len(self._latencies), 1)


# ▌ HUD OVERLAY

# State → (label_text, BGR_color)
_STATE_STYLE = {
    DriveState.DRIVE: ("DRIVE",  (0, 220, 0)),
    DriveState.STOP:  ("STOP",   (0, 0, 255)),
    DriveState.LOST:  ("LOST",   (0, 165, 255)),
}


def _draw_hud(display, state, left, right, steering, latency_ms, fps):
    h, w  = display.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # ── Top banner 
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    label, color = _STATE_STYLE.get(state, ("???", (200, 200, 200)))
    cv2.putText(display, f"[{label}]", (8, 32), font, 0.85, color, 2)

    cv2.putText(display, f"L:{left:+4d}  R:{right:+4d}",
                (170, 32), font, 0.72, (255, 255, 255), 2)

    steer_str = f"Steer:{steering:+6.1f}d" if steering is not None else "Steer:  LOST"
    steer_col = (200, 200, 255) if steering is not None else (0, 100, 255)
    cv2.putText(display, steer_str, (420, 32), font, 0.68, steer_col, 2)

    # ── Bottom-right: perf stats 
    cv2.putText(display, f"{fps:.1f} FPS  {latency_ms:.1f}ms",
                (w - 220, h - 10), font, 0.6, (100, 255, 100), 1)

    # ── Steering bar (visual gauge) 
    bar_cx = w // 2
    bar_y  = h - 18
    bar_hw = 80   # half-width of the bar
    cv2.rectangle(display, (bar_cx - bar_hw, bar_y - 6),
                  (bar_cx + bar_hw, bar_y + 6), (50, 50, 50), -1)

    if steering is not None:
        indicator_x = int(bar_cx + np.clip(steering / 35.0, -1.0, 1.0) * bar_hw)
        bar_color   = (0, 200, 255) if abs(steering) < 15 else (0, 80, 255)
        cv2.rectangle(display, (bar_cx, bar_y - 6),
                      (indicator_x, bar_y + 6), bar_color, -1)

    cv2.line(display, (bar_cx, bar_y - 10), (bar_cx, bar_y + 10), (200, 200, 200), 1)


# ▌ MAIN LOOP

def main():
    print("=" * 62)
    print("  ZENITY ROV  |  v3.0  |  Initialising…")
    print("=" * 62)

    ai_brain      = ZenityBrain()
    state_machine = StateMachine()
    perf          = PerfTracker()
    stop_event    = threading.Event()

    esp_addr = (ESP32_IP, ESP32_PORT)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setblocking(False)

    # ── Graceful shutdown 
    def _shutdown(sig=None, frame=None):
        print("\n[main] Shutdown — sending STOP to ESP32 …")
        stop_event.set()
        for _ in range(3):                      # send 3× for reliability
            _send_udp(udp_sock, esp_addr, "0,0\n")
            time.sleep(0.05)
        udp_sock.close()
        cv2.destroyAllWindows()
        print("[main] Clean exit.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Threads 
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

    last_cmd_time = 0.0
    cmd_interval  = 1.0 / COMMAND_HZ
    frame_count   = 0
    left_pwm      = 0
    right_pwm     = 0

    while not stop_event.is_set():

        # ── Frame acquisition 
        try:
            frame = frame_queue.get(timeout=CAMERA_TIMEOUT_S)
        except queue.Empty:
            print("[main] ⚠  No frame — camera down?")
            _send_udp(udp_sock, esp_addr, "0,0\n")   # safe default
            continue

        t0 = time.perf_counter()

        # ── AI inference 
        display, steering, stop_detected = ai_brain.process_frame(frame)
        ai_ms = (time.perf_counter() - t0) * 1000

        if display is None:
            continue

        # ── State machine → PWM 
        left_pwm, right_pwm = state_machine.update(stop_detected, steering)

        # ── Rate-limited UDP send 
        now = time.perf_counter()
        if now - last_cmd_time >= cmd_interval:
            last_cmd_time = now
            _send_udp(udp_sock, esp_addr, f"{left_pwm},{right_pwm}\n")

            frame_count += 1
            if frame_count % 10 == 0:
                state_str = state_machine.state
                steer_str = f"{steering:+.1f}°" if steering is not None else "N/A"
                print(
                    f"[{state_str:<5}]  L:{left_pwm:+4d}  R:{right_pwm:+4d}"
                    f"  steer:{steer_str:>8}  AI:{ai_ms:5.1f}ms"
                    f"  FPS:{perf.fps:4.1f}"
                )

        # ── Dashboard 
        perf.tick(ai_ms)
        _draw_hud(display, state_machine.state,
                  left_pwm, right_pwm, steering, ai_ms, perf.fps)

        cv2.imshow("Zenity ROV  —  v3.0", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            # Manual e-stop: press SPACE to immediately send 0,0
            print("[main] Manual E-STOP via SPACE key")
            _send_udp(udp_sock, esp_addr, "0,0\n")

    _shutdown()


if __name__ == "__main__":
    main()