import cv2
import easyocr
import re
import time

print("Loading EasyOCR Engine...")
reader = easyocr.Reader(['en'], gpu=False)

def read_speed_limit(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find {image_path}")
        return
    
    start_time = time.time()
    
    # Pre-processing to boost contrast
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    print("Scanning image for text...")
    results = reader.readtext(thresh)

    print("\n--- RAW OCR OUTPUT (Debugging) ---")
    detected_speed = None

    for (bbox, text, prob) in results:
        print(f"AI Saw: '{text}' (Confidence: {prob:.2f})")
        
        # THE FIX: Find any 2 or 3 digits anywhere in the string, ignoring the garbage around it.
        match = re.search(r'\d{2,3}', text)
        
        # THE FIX: Dropped the confidence threshold to 0.01 for this noisy test
        if match and prob > 0.01: 
            detected_speed = int(match.group())
            print(f"✅ Speed Limit Extracted: {detected_speed}")
            
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(img, str(detected_speed), (top_left[0], top_left[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break

    latency = (time.time() - start_time) * 1000
    print(f"\nPipeline Latency: {latency:.1f}ms")

    if detected_speed is None:
        print("❌ Still no valid numbers found.")

    cv2.imshow("OCR Output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

read_speed_limit("speed_sign.jpg")