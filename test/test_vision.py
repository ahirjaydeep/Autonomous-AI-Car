import requests
import kornia_rs as K
import cv2
import time

# 1. REPLACE THIS with the exact IP address shown on your phone screen
# Make sure you keep the "/video" part at the very end!
STREAM_URL = "http://10.134.160.163:8080/video"

# 2. Initialize the Rust-based decoder
decoder = K.ImageDecoder()

print(f"Connecting to phone stream at {STREAM_URL}...")
response = requests.get(STREAM_URL, stream=True)
bytes_buffer = b''

print("Stream connected! Press 'q' on your keyboard to quit.")

# 3. The Video Loop
for chunk in response.iter_content(chunk_size=4096):
    bytes_buffer += chunk
    
    # JPEGs always start with \xff\xd8 and end with \xff\xd9
    a = bytes_buffer.find(b'\xff\xd8')
    b = bytes_buffer.find(b'\xff\xd9')
    
    if a != -1 and b != -1:
        start_time = time.time()
        
        # Isolate the exact JPEG bytes
        jpg_bytes = bytes_buffer[a:b+2]
        bytes_buffer = bytes_buffer[b+2:] # Clear buffer for the next frame
        
        # --- THE KORNIA-RS TEST ---
        # Decode the raw bytes using Rust
        decoded_img = decoder.decode(jpg_bytes)
        
        latency = (time.time() - start_time) * 1000
        print(f"kornia-rs decode time: {latency:.2f} ms | Shape: {decoded_img.shape}")
        
        # --- HUMAN VISUALIZATION ---
        # kornia-rs decodes into RGB, but OpenCV expects BGR to show it on a Mac screen.
        # (Note: In the final AI code, we skip OpenCV completely for maximum speed).
        img_bgr = cv2.cvtColor(decoded_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Zenity ROV - Camera Test", img_bgr)
        
        # Press 'q' to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()