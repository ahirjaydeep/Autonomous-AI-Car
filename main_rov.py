import cv2
import time
import queue
import threading
import requests
import kornia_rs as K
import torch
from ai_engine import ZenityBrain

# --- SYSTEM CONFIGURATION ---
PHONE_IP = "10.134.160.163"     # YOUR PHONE IP
ESP32_IP = "192.168.x.x"     # WE WILL GET THIS LATER
STREAM_URL = f"http://{PHONE_IP}:8080/shot.jpg"

BASE_SPEED = 180  # Speed of the car (0 to 255)

# Initialize the AI Brain
ai_brain = ZenityBrain()
decoder = K.ImageDecoder()

def steering_to_tank(steering_deg):
    """Converts a steering angle to left and right wheel speeds."""
    t = steering_deg / 35.0  # Normalize max steering to 1.0
    t = max(-1.0, min(1.0, t))
    
    left = int(BASE_SPEED * (1.0 + t))
    right = int(BASE_SPEED * (1.0 - t))
    return max(0, min(255, left)), max(0, min(255, right))

def camera_thread(q):
    """Background thread that aggressively pulls the newest frame."""
    session = requests.Session()
    while True:
        try:
            resp = session.get(STREAM_URL, timeout=1.0)
            
            # Fast Rust Decode
            decoded = decoder.decode(resp.content)
            img_rgb = torch.from_dlpack(decoded).numpy()
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Drop old frames, keep only the absolute newest
            if q.full():
                q.get_nowait()
            q.put(img_bgr)
            
        except Exception as e:
            time.sleep(0.1)

def main():
    print("Starting Zenity ROV System...")
    
    # Start the camera thread
    frame_queue = queue.Queue(maxsize=2)
    threading.Thread(target=camera_thread, args=(frame_queue,), daemon=True).start()
    
    esp_session = requests.Session()
    last_command_time = 0

    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue
            
        frame = frame_queue.get()
        t_start = time.time()
        
        # --- SEND TO AI BRAIN ---
        display, steering, stop_detected = ai_brain.process_frame(frame)
        
        # --- SEND TO ESP32 ---
        now = time.time()
        if now - last_command_time > 0.1: # Max 10 commands per second
            last_command_time = now
            
            try:
                if stop_detected:
                    print("🛑 AI COMMAND: BRAKE")
                    # esp_session.get(f"http://{ESP32_IP}/drive?left=0&right=0", timeout=0.1)
                elif steering is not None:
                    left_pwm, right_pwm = steering_to_tank(steering)
                    print(f"✅ AI COMMAND: DRIVE | L:{left_pwm} R:{right_pwm}")
                    # esp_session.get(f"http://{ESP32_IP}/drive?left={left_pwm}&right={right_pwm}", timeout=0.1)
            except requests.exceptions.RequestException:
                pass # Fire and forget, don't let Wi-Fi lag slow down the Mac

        # --- CALCULATE & DISPLAY LATENCY ---
        latency = (time.time() - t_start) * 1000
        cv2.putText(display, f"System Latency: {latency:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Zenity ROV Dashboard", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("System Shutdown.")

if __name__ == "__main__":
    main()