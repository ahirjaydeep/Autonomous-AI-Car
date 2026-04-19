import cv2
import numpy as np
from ultralytics import YOLO

class PIDController:
    def __init__(self, kp, ki, kd, max_out):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return max(-self.max_out, min(self.max_out, output))

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

class RovBrain:
    def __init__(self):
        print("Loading YOLO11 NANO to Apple M3 (MPS) for Max Speed...")
        self.yolo = YOLO("yolo11n.pt")
        self.yolo.to("mps")
        
        self.tracked_classes = [0, 2, 9, 11] # Person, Car, Light, Stop Sign
        self.class_names = {0: "PERSON", 2: "CAR", 9: "LIGHT", 11: "STOP SIGN"}

        self.pid = PIDController(kp=0.2, ki=0.0, kd=0.05, max_out=35.0)
        
        # Simplified lighting fix to save milliseconds
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.brake_area_threshold = 0.04 

    def process_frame(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            return None, None, False

        h, w = frame_bgr.shape[:2]
        frame_area = w * h
        display = frame_bgr.copy()
        stop_detected = False
        steering_angle = None

        # --- 1. YOLO11 NANO INFERENCE ---
        try:
            results = self.yolo(frame_bgr, classes=self.tracked_classes, conf=0.45, verbose=False)
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    box_area = (x2 - x1) * (y2 - y1)
                    area_ratio = box_area / frame_area
                    
                    label = f"{self.class_names.get(cls_id, 'OBJ')} {conf:.2f} (Dist: {area_ratio:.2f})"
                    
                    if cls_id == 11 and area_ratio > self.brake_area_threshold:
                        stop_detected = True
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(display, "BRAKING!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 100, 0), 2)
                        cv2.putText(display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        except Exception as e:
            pass

        # --- 2. FAST DUAL-LANE DETECTION ---
        try:
            roi_y = int(h * 0.50) 
            roi = frame_bgr[roi_y:h, :]
            roi_h, roi_w = roi.shape[:2]
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = self.clahe.apply(gray)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            mid_x = roi_w // 2
            left_mask = mask[:, :mid_x]
            right_mask = mask[:, mid_x:]

            left_cx, right_cx = None, None

            l_contours, _ = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if l_contours:
                c = max(l_contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 100:
                    M = cv2.moments(c)
                    if M["m00"] != 0: left_cx = int(M["m10"]/M["m00"])

            r_contours, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if r_contours:
                c = max(r_contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 100:
                    M = cv2.moments(c)
                    if M["m00"] != 0: right_cx = int(M["m10"]/M["m00"]) + mid_x 

            # --- 3. STEERING LOGIC ---
            center_x = roi_w // 2
            target_x = center_x
            lane_width_estimate = 300 

            if left_cx is not None and right_cx is not None:
                target_x = (left_cx + right_cx) // 2
            elif left_cx is not None:
                target_x = left_cx + (lane_width_estimate // 2)
            elif right_cx is not None:
                target_x = right_cx - (lane_width_estimate // 2)
            else:
                self.pid.reset()
                return display, None, stop_detected

            error = target_x - center_x
            steering_angle = self.pid.update(error)
            
            # Draw tracking lines
            cv2.line(display, (center_x, h), (target_x, roi_y + 50), (0, 255, 0), 3)
            
        except Exception as e:
            self.pid.reset()

        return display, steering_angle, stop_detected