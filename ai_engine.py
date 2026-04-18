import cv2
import numpy as np
import torch
from ultralytics import YOLO

class PIDController:
    """Advanced PID Controller for smooth steering."""
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

class ZenityBrain:
    """The Upgraded AI Engine: Multi-Class YOLO and Dual-Lane Tracking."""
    def __init__(self):
        print("Loading YOLO11 Small to Apple M3 (MPS)...")
        # UPGRADE: Using the heavier, more accurate 'Small' model
        self.yolo = YOLO("yolo11s.pt")
        self.yolo.to("mps")
        
        # UPGRADE: Track People(0), Cars(2), Traffic Lights(9), Stop Signs(11)
        self.tracked_classes = [0, 2, 9, 11]
        
        # Class names for the dashboard overlay
        self.class_names = {0: "PERSON", 2: "CAR", 9: "LIGHT", 11: "STOP SIGN"}

        self.pid = PIDController(kp=0.2, ki=0.0, kd=0.05, max_out=35.0)

        # Advanced Vision Tool for lighting
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def process_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        display = frame_bgr.copy()
        stop_detected = False
        steering_angle = None

        # ---------------------------------------------------------
        # 1. MULTI-CLASS YOLO INFERENCE
        # ---------------------------------------------------------
        results = self.yolo(frame_bgr, classes=self.tracked_classes, conf=0.45, verbose=False)
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                label = f"{self.class_names.get(cls_id, 'OBJECT')} {conf:.2f}"
                
                # If it's a Stop Sign, trigger the brake flag
                if cls_id == 11:
                    stop_detected = True
                    color = (0, 0, 255) # Red
                else:
                    color = (255, 100, 0) # Blue for cars/people
                
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if stop_detected:
            cv2.putText(display, "EMERGENCY OVERRIDE", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # ---------------------------------------------------------
        # 2. DUAL-LANE DETECTION PIPELINE
        # ---------------------------------------------------------
        roi_y = int(h * 0.50) # Look at bottom 50%
        roi = frame_bgr[roi_y:h, :]
        roi_h, roi_w = roi.shape[:2]
        
        # Convert to grayscale and fix bad lighting/glare
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Adaptive Threshold (Finds lines even in shadows)
        mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 15, 3)

        # UPGRADE: Split the screen in half to find Left vs Right lines
        mid_x = roi_w // 2
        left_mask = mask[:, :mid_x]
        right_mask = mask[:, mid_x:]

        left_cx = None
        right_cx = None

        # --- Find Left Line ---
        l_contours, _ = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if l_contours:
            c = max(l_contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 150:
                M = cv2.moments(c)
                if M["m00"] != 0: 
                    left_cx = int(M["m10"]/M["m00"])
                    cv2.circle(display, (left_cx, roi_y + int(M["m01"]/M["m00"])), 8, (255, 0, 0), -1)

        # --- Find Right Line ---
        r_contours, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if r_contours:
            c = max(r_contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 150:
                M = cv2.moments(c)
                if M["m00"] != 0: 
                    # Add mid_x because this contour is from the right half of the image
                    right_cx = int(M["m10"]/M["m00"]) + mid_x 
                    cv2.circle(display, (right_cx, roi_y + int(M["m01"]/M["m00"])), 8, (255, 0, 255), -1)

        # ---------------------------------------------------------
        # 3. STEERING TARGET LOGIC
        # ---------------------------------------------------------
        center_x = roi_w // 2
        target_x = center_x
        lane_width_estimate = 300 # Pixels between left and right lines (tune this on track)

        if left_cx is not None and right_cx is not None:
            # We see both lines. Drive exactly in the middle.
            target_x = (left_cx + right_cx) // 2
            cv2.line(display, (left_cx, h - 20), (right_cx, h - 20), (0, 255, 255), 2)
            
        elif left_cx is not None:
            # We only see the left line. Guess where the center is.
            target_x = left_cx + (lane_width_estimate // 2)
            cv2.putText(display, "RIGHT LINE LOST", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
        elif right_cx is not None:
            # We only see the right line. Guess where the center is.
            target_x = right_cx - (lane_width_estimate // 2)
            cv2.putText(display, "LEFT LINE LOST", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
        else:
            self.pid.reset()
            cv2.putText(display, "TOTAL LANE LOSS", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            return display, None, stop_detected

        # Calculate PID error and output
        error = target_x - center_x
        steering_angle = self.pid.update(error)

        # Draw the target line
        cv2.line(display, (center_x, h), (target_x, roi_y + 50), (0, 255, 0), 3)

        return display, steering_angle, stop_detected