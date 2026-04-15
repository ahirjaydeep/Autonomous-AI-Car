import requests
import kornia_rs as K
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import time

# 1. Enter the Local IP Address from your phone's "IP Webcam" app
STREAM_URL = "http://172.28.251.234:8080/video"

print("Loading YOLO26 Nano...")
# 2. Load the Model and push it to the Mac M3 Neural Engine
model = YOLO("yolo26n.pt")
model.to("mps") 

# 3. Initialize the Rust-based zero-copy decoder
decoder = K.ImageDecoder()

print(f"Connecting to Samsung A51 stream at {STREAM_URL}...")
response = requests.get(STREAM_URL, stream=True)
bytes_buffer = b''

print("Stream connected! Look for the popup window. Press 'q' to quit.")

# 4. The Infinite AI Loop
for chunk in response.iter_content(chunk_size=4096):
    bytes_buffer += chunk
    
    # JPEGs always start with \xff\xd8 and end with \xff\xd9
    a = bytes_buffer.find(b'\xff\xd8')
    b = bytes_buffer.find(b'\xff\xd9')
    
    # If we have captured a full video frame
    if a != -1 and b != -1:
        start_time = time.time()
        
        # Isolate the exact JPEG bytes
        jpg_bytes = bytes_buffer[a:b+2]
        bytes_buffer = bytes_buffer[b+2:] 
        
        # --- KORNIA-RS DECODE ---
        decoded_dlpack = decoder.decode(jpg_bytes)
        img_tensor = torch.from_dlpack(decoded_dlpack)
        
        # Convert to NumPy array so OpenCV can physically draw it on your Mac screen
        img_rgb = img_tensor.numpy()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # --- NMS-FREE INFERENCE ---
        results = model(img_bgr, verbose=False)
        
        # Calculate Latency
        latency = (time.time() - start_time) * 1000
        print(f"Frame Processed in: {latency:.1f} ms")
        
        # --- VISUALIZATION (Adding Eyes) ---
        # .plot() tells YOLO to draw the colored boxes and labels on the image
        annotated_frame = results[0].plot()
        
        # Show the video stream in a Mac window
        cv2.imshow("Zenity ROV - AI Vision", annotated_frame)
        
        # This keeps the window open and listens for you to press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close the window cleanly when done
cv2.destroyAllWindows()