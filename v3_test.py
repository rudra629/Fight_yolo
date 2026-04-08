import cv2
import numpy as np
import math
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict, deque

print("Booting up the Center-of-Mass Fight Detection Engine (v3 Test)...")
model = YOLO('yolov8n-pose.pt') 

# ==========================================
# CONFIGURATION & THRESHOLDS (TUNED FOR TORSOS)
# ==========================================
# Torsos move slower than wrists! A body lunge of 20 pixels is massive.
TORSO_VELOCITY_THRESHOLD = 20 
PROXIMITY_THRESHOLD = 180 

# Tracking dictionaries (Now tracking the chest/stomach, not the wrists)
body_history = defaultdict(lambda: deque(maxlen=3)) 
fight_buffer = deque(maxlen=10) 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture('newfi13.avi')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\nSystem Armed. Press 'q' to quit.")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calc_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def draw_hud(frame, safe, active_count):
    """Draws a sleek, semi-transparent security camera UI"""
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (350, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"SYS: CENTER-OF-MASS ENGINE v3.0", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"TIME: {time_str}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"TARGETS TRACKED: {active_count}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"TORSO VELOCITY LIMIT: {TORSO_VELOCITY_THRESHOLD}px", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"PROXIMITY LIMIT: {PROXIMITY_THRESHOLD}px", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    length = 40
    color = (0, 255, 0) if safe else (0, 0, 255)
    thickness = 4
    cv2.line(frame, (10, 10), (10 + length, 10), color, thickness)
    cv2.line(frame, (10, 10), (10, 10 + length), color, thickness)
    cv2.line(frame, (w - 10, 10), (w - 10 - length, 10), color, thickness)
    cv2.line(frame, (w - 10, 10), (w - 10, 10 + length), color, thickness)
    cv2.line(frame, (10, h - 10), (10 + length, h - 10), color, thickness)
    cv2.line(frame, (10, h - 10), (10, h - 10 - length), color, thickness)
    cv2.line(frame, (w - 10, h - 10), (w - 10 - length, h - 10), color, thickness)
    cv2.line(frame, (w - 10, h - 10), (w - 10, h - 10 - length), color, thickness)

# ==========================================
# MAIN VIDEO LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]

    # Added conf=0.50 to ignore blurry background objects
    results = model.track(frame, persist=True, classes=0, conf=0.50, verbose=False)
    
    is_fighting = False
    annotated_frame = results[0].plot() 
    active_people = []

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xywh.cpu().numpy() 

        for i, track_id in enumerate(track_ids):
            box = boxes[i]
            # Tracking the absolute center of the bounding box (The Torso)
            center_x, center_y = box[0], box[1] 
            
            history = body_history[track_id]
            body_speed = 0
            
            if len(history) > 0:
                prev_center = history[-1]
                body_speed = calc_distance((center_x, center_y), prev_center)
            
            history.append((center_x, center_y))
            
            # Filter out crazy YOLO teleportation glitches
            if body_speed > 100:
                body_speed = 0
            
            active_people.append({
                'id': track_id, 'center': (center_x, center_y), 'speed': body_speed
            })

        # 2. COLLISION CHECK (Torso-Based)
        if len(active_people) >= 2:
            for i in range(len(active_people)):
                for j in range(i + 1, len(active_people)):
                    p1, p2 = active_people[i], active_people[j]
                    
                    distance = calc_distance(p1['center'], p2['center'])
                    
                    # Did one of their bodies violently lunge?
                    aggressive_movement = (p1['speed'] > TORSO_VELOCITY_THRESHOLD) or (p2['speed'] > TORSO_VELOCITY_THRESHOLD)
                    
                    if distance < PROXIMITY_THRESHOLD and aggressive_movement:
                        is_fighting = True
                        break 

    fight_buffer.append(1 if is_fighting else 0)
    
    # Require 4 frames of continuous violence to trigger
    is_safe = sum(fight_buffer) < 4
    
    draw_hud(annotated_frame, is_safe, len(active_people))
    
    if not is_safe:
        cv2.putText(annotated_frame, "🚨 CRITICAL: FIGHT DETECTED! 🚨", (w//2 - 350, 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
        cv2.rectangle(annotated_frame, (0,0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0,0,255), 15)
    else:
        cv2.putText(annotated_frame, "STATUS: SCENE SAFE", (w//2 - 150, 80), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Enterprise Threat Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()