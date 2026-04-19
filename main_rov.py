"""
================================================================================
 ZENITY ROV — main_rov.py  |  "The Nervous System"
 Version: 2.0 (Production-Grade)
================================================================================
 Responsibilities:
   - Camera thread: aggressive pull from IP Webcam /shot.jpg  (no buffer bloat)
   - UDP command socket: fires L,R PWM strings to ESP32 at ≤10 ms overhead
   - Heartbeat thread: sends a keep-alive every 500 ms; ESP32 can E-STOP if
     it goes silent (implement a watchdog timer on the C++ side)
   - Drive state machine: DRIVE → STOP → RESUME  (debounced, clean transitions)
   - Graceful shutdown: sends STOP command before Python exits

 UDP Packet format (UTF-8):
   Normal drive →  "180,150\n"
   Full stop    →  "0,0\n"
   Heartbeat    →  "PING\n"

 ESP32 C++ snippet (for reference):
   void loop() {
     int len = udp.parsePacket();
     if (len) {
       char buf[32]; udp.read(buf, len); buf[len] = 0;
       if (strcmp(buf, "PING\n") == 0) { lastPing = millis(); return; }
       int l, r; sscanf(buf, "%d,%d", &l, &r);
       analogWrite(MOTOR_L, l); analogWrite(MOTOR_R, r);
       lastPing = millis();
     }
     // Watchdog: if no packet for 1 s, E-STOP
     if (millis() - lastPing > 1000) { analogWrite(MOTOR_L, 0); analogWrite(MOTOR_R, 0); }
   }
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

try:
    import kornia_rs as K
    import torch
    _USE_KORNIA = True
except ImportError:
    _USE_KORNIA = False
    print("[main] kornia_rs not found — falling back to cv2.imdecode")

from ai_engine5 import ZenityBrain



# CONFIGURATION


PHONE_IP    = "10.48.167.163"      # IP Webcam phone
ESP32_IP    = "10.48.167.62"       # Fill in after flashing ESP32
ESP32_PORT  = 4210                # UDP port — match with C++ code

STREAM_URL  = f"http://{PHONE_IP}:8080/shot.jpg"

BASE_SPEED          = 90   # 0-255 — overall drive speed
MAX_STOP_DURATION   = 3.0   # seconds to hold brake after stop detected
COMMAND_HZ          = 10    # max motor commands per second  (100 ms gate)
HEARTBEAT_HZ        = 2     # keep-alive pings per second to ESP32


# DRIVE STATE MACHINE


class DriveState:
    DRIVE  = "DRIVE"
    STOP   = "STOP"

class StateMachine:
    """
    Converts raw per-frame AI outputs into clean, debounced motor commands.

    Transitions:
      DRIVE  →  STOP   : stop_detected == True
      STOP   →  DRIVE  : stop has been held for MAX_STOP_DURATION seconds
    """

    def __init__(self):
        self.state      = DriveState.DRIVE
        self._stop_ts   = None           # timestamp we entered STOP

    def update(self, stop_detected: bool, steering: float | None):
        """
        Returns (left_pwm, right_pwm) based on current state.
        """
        if self.state == DriveState.DRIVE:
            if stop_detected:
                self.state    = DriveState.STOP
                self._stop_ts = time.time()
                return 0, 0

            if steering is None:
                # Lane lost — go straight and hope for the best
                return BASE_SPEED, BASE_SPEED

            left, right = _steering_to_tank(steering)
            return left, right

        elif self.state == DriveState.STOP:
            elapsed = time.time() - self._stop_ts
            if elapsed >= MAX_STOP_DURATION:
                print(f"[StateMachine] Stop held for {elapsed:.1f}s → resuming DRIVE")
                self.state = DriveState.DRIVE
            return 0, 0

        return 0, 0   # fallback safety



# HELPERS


def _steering_to_tank(steering_deg: float):
    """
    Maps a PID steering angle to differential (tank) drive PWM values.
    Upgraded for 4WD Skid-Steering at low base speeds.
    """
    # 1. Sensitivity limit (lower divisor = sharper steering response)
    t = max(-1.0, min(1.0, steering_deg / 15.0))
    
    # 2. Turn Power Factor 
    # This guarantees enough torque to skid the tires even if BASE_SPEED is 90
    TURN_POWER = 110
    
    # 3. Additive Mixing 
    # (Note: If the car turns the wrong way again, just swap the + and - signs below)
    left  = int(BASE_SPEED + (t * TURN_POWER))
    right = int(BASE_SPEED - (t * TURN_POWER))
    
    # 4. Clamp to -255 and 255. 
    # Allowing negative numbers means the inner wheels will physically spin 
    # backwards on sharp turns, forcing the car to pivot!
    return max(-255, min(255, left)), max(-255, min(255, right))


def _send_udp(sock: socket.socket, addr: tuple, payload: str):
    """Fire-and-forget UDP send.  Never blocks the main loop."""
    try:
        sock.sendto(payload.encode(), addr)
    except OSError:
        pass



# CAMERA THREAD


def camera_thread(q: queue.Queue, stop_event: threading.Event):
    """
    Pulls /shot.jpg from the phone in a tight loop.
    Only keeps the NEWEST frame — old frames are discarded immediately.
    Uses kornia_rs Rust decoder if available (faster), otherwise cv2.imdecode.
    """
    session = requests.Session()
    session.headers.update({"Connection": "keep-alive"})

    decoder = K.ImageDecoder() if _USE_KORNIA else None
    backoff = 0.05   # seconds to wait after a failure (exponential up to 1 s)

    while not stop_event.is_set():
        try:
            resp = session.get(STREAM_URL, timeout=1.5)
            resp.raise_for_status()

            # ── Decode 
            if _USE_KORNIA and decoder is not None:
                decoded  = decoder.decode(resp.content)
                img_rgb  = torch.from_dlpack(decoded).numpy()
                img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            else:
                arr     = np.frombuffer(resp.content, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                raise ValueError("Decoded frame is None")

            # ── Drop stale frames — only ever hold 1 
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put(img_bgr)

            backoff = 0.05   # reset backoff on success

        except Exception as exc:
            print(f"[Camera] Error: {exc}  — retry in {backoff:.2f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 1.0)   # cap at 1 second



# HEARTBEAT THREAD


def heartbeat_thread(sock: socket.socket, addr: tuple, stop_event: threading.Event):
    """
    Sends "PING\\n" to the ESP32 at HEARTBEAT_HZ.
    If this thread goes silent (Python crash), the ESP32 watchdog will E-STOP.
    """
    interval = 1.0 / HEARTBEAT_HZ
    while not stop_event.is_set():
        _send_udp(sock, addr, "PING\n")
        time.sleep(interval)



# MAIN LOOP


def main():
    print("=" * 60)
    print("  ZENITY ROV  |  v2.0  |  Initialising…")
    print("=" * 60)

    # ── Subsystems 
    ai_brain    = ZenityBrain()
    state_machine = StateMachine()
    stop_event  = threading.Event()

    # ── UDP socket (shared by main loop + heartbeat thread)
    esp_addr = (ESP32_IP, ESP32_PORT)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setblocking(False)

    # ── Graceful shutdown handler
    def _shutdown(sig=None, frame=None):
        print("\n[main] Shutdown signal received — sending STOP to ESP32…")
        stop_event.set()
        _send_udp(udp_sock, esp_addr, "0,0\n")   # motors off
        time.sleep(0.2)
        udp_sock.close()
        cv2.destroyAllWindows()
        print("[main] Clean exit.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Threads 
    frame_queue = queue.Queue(maxsize=2)

    t_cam = threading.Thread(
        target=camera_thread, args=(frame_queue, stop_event), daemon=True
    )
    t_hb = threading.Thread(
        target=heartbeat_thread, args=(udp_sock, esp_addr, stop_event), daemon=True
    )
    t_cam.start()
    t_hb.start()

    print("[main] Camera + heartbeat threads started. Waiting for first frame…")

    last_cmd_time  = 0.0
    cmd_interval   = 1.0 / COMMAND_HZ
    frame_count    = 0

    while not stop_event.is_set():
        # ── Frame acquisition
        try:
            frame = frame_queue.get(timeout=2.0)
        except queue.Empty:
            print("[main] ⚠  No frame in 2 s — camera feed down?")
            continue

        t0 = time.perf_counter()

        # ── AI inference 
        display, steering, stop_detected = ai_brain.process_frame(frame)

        ai_ms = (time.perf_counter() - t0) * 1000

        if display is None:
            continue

        # ── State machine → motor command 
        left_pwm, right_pwm = state_machine.update(stop_detected, steering)

        # ── Rate-limited UDP send 
        now = time.perf_counter()
        if now - last_cmd_time >= cmd_interval:
            last_cmd_time = now
            payload = f"{left_pwm},{right_pwm}\n"
            _send_udp(udp_sock, esp_addr, payload)

            # Console log (only every 10th frame to avoid spam)
            frame_count += 1
            if frame_count % 10 == 0:
                state_label = "🛑 STOP" if state_machine.state == DriveState.STOP else "✅ DRIVE"
                steer_label = f"{steering:+.1f}°" if steering is not None else "N/A"
                print(
                    f"[{state_label}]  L:{left_pwm:3d}  R:{right_pwm:3d}"
                    f"  steer:{steer_label}  AI:{ai_ms:.1f}ms"
                )

        # ── Dashboard overlay
        total_ms = (time.perf_counter() - t0) * 1000

        _draw_hud(display, state_machine.state, left_pwm, right_pwm, steering, total_ms)

        cv2.imshow("Zenity ROV Dashboard", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    _shutdown()



# HUD OVERLAY


def _draw_hud(display, state, left, right, steering, latency_ms):
    """Draws a clean informational overlay on the dashboard frame."""
    h, w = display.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # Semi-transparent black banner at top
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)

    state_color = (0, 0, 255) if state == DriveState.STOP else (0, 255, 0)
    cv2.putText(display, f"STATE: {state}", (10, 30), font, 0.8, state_color, 2)
    cv2.putText(display, f"L:{left:3d}  R:{right:3d}", (230, 30), font, 0.7, (255, 255, 255), 2)

    steer_str = f"Steer: {steering:+.1f} deg" if steering is not None else "Steer: LOST"
    cv2.putText(display, steer_str, (430, 30), font, 0.7, (200, 200, 255), 2)
    cv2.putText(display, f"{latency_ms:.1f} ms", (w - 120, 30), font, 0.7, (100, 255, 100), 2)




if __name__ == "__main__":
    main()