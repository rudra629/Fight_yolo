# WORKING TITLE: v1.py
import cv2
import numpy as np
import math
import time
import threading
import requests
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict, deque

print("Booting up the Enterprise Kinematic Fight Detection Engine...")
model = YOLO('yolov8n-pose.pt') 

# ==========================================
# CONFIGURATION & THRESHOLDS
# ==========================================
VELOCITY_THRESHOLD = 40 
PROXIMITY_THRESHOLD = 200 

# ==========================================
# CLOUD API SETTINGS (From app.py)
# ==========================================
BEACON_ID = "ab907856-3412-3412-3412-341278563412"
DEVICE_ID = "AI-VIDEO-MONITORING-EDGE"
BACKEND_URL = "https://resq-server.onrender.com/api/violence-detcted/"

LAST_API_CALL = 0
API_COOLDOWN = 10.0 # Wait 10 seconds between API calls so we don't spam the Render server

# Tracking dictionaries
wrist_history = defaultdict(lambda: deque(maxlen=3)) 
fight_buffer = deque(maxlen=10) 

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture('newfi11.avi')
# Force high-speed resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\nSystem Armed. Press 'q' to quit.")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calc_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def send_alert_worker(frame):
    """Encodes the frame with skeletons to JPEG and uploads to Render API"""
    try:
        print("\n[NETWORK] 🚨 ENCODING EVIDENCE IMAGE...")
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: 
            print("[NETWORK] ❌ Failed to encode image.")
            return

        timestamp = int(time.time())
        unique_filename = f"fight_alert_{timestamp}.jpg"

        data = {
            'beacon_id': BEACON_ID,
            'confidence_score': "0.95", # YOLO kinematic math is highly confident when triggered
            'description': "Kinematic Physical Fight Detected by YOLOv8",
            'device_id': DEVICE_ID
        }
        
        # Attach the image file just like in app.py
        files = {'images': (unique_filename, buffer.tobytes(), 'image/jpeg')}

        print(f"[NETWORK] 🚀 UPLOADING TO: {BACKEND_URL}")
        response = requests.post(BACKEND_URL, data=data, files=files, timeout=10)
        print(f"[NETWORK] ✅ API Response Status: {response.status_code}\n")
    except Exception as e:
        print(f"[NETWORK] ❌ Upload Error: {e}\n")

def draw_hud(frame, safe, active_count):
    """Draws a sleek, semi-transparent security camera UI"""
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (350, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"SYS: KINEMATIC ENGINE v2.0", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"TIME: {time_str}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"TARGETS TRACKED: {active_count}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"VELOCITY LIMIT: {VELOCITY_THRESHOLD}px", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
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

    results = model.track(frame, persist=True, classes=0, verbose=False)
    
    is_fighting = False
    annotated_frame = results[0].plot() 
    active_people = []

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        keypoints = results[0].keypoints.xy.cpu().numpy()
        boxes = results[0].boxes.xywh.cpu().numpy() 

        for i, track_id in enumerate(track_ids):
            kpts = keypoints[i]
            center_x, center_y = boxes[i][0], boxes[i][1]
            
            if len(kpts) > 10 and kpts[9][0] != 0 and kpts[10][0] != 0:
                l_wrist, r_wrist = kpts[9], kpts[10]
                history = wrist_history[track_id]
                
                max_speed = 0
                if len(history) > 0:
                    prev_l_wrist, prev_r_wrist = history[-1]
                    l_speed = calc_distance(l_wrist, prev_l_wrist)
                    r_speed = calc_distance(r_wrist, prev_r_wrist)
                    max_speed = max(l_speed, r_speed)
                
                history.append((l_wrist, r_wrist))
                
                active_people.append({
                    'id': track_id, 'center': (center_x, center_y), 'speed': max_speed
                })

        if len(active_people) >= 2:
            for i in range(len(active_people)):
                for j in range(i + 1, len(active_people)):
                    p1, p2 = active_people[i], active_people[j]
                    
                    distance = calc_distance(p1['center'], p2['center'])
                    aggressive_movement = (p1['speed'] > VELOCITY_THRESHOLD) or (p2['speed'] > VELOCITY_THRESHOLD)
                    
                    if distance < PROXIMITY_THRESHOLD and aggressive_movement:
                        is_fighting = True
                        break 

    fight_buffer.append(1 if is_fighting else 0)
    
    is_safe = sum(fight_buffer) < 4
    
    draw_hud(annotated_frame, is_safe, len(active_people))
    
    if not is_safe:
        cv2.putText(annotated_frame, "🚨 CRITICAL: FIGHT DETECTED! 🚨", (w//2 - 350, 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
        cv2.rectangle(annotated_frame, (0,0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0,0,255), 15)
        
        # API Trigger (With Cooldown and Frame Copy)
        current_time = time.time()
        if current_time - LAST_API_CALL > API_COOLDOWN:
            LAST_API_CALL = current_time
            # We pass a copy of the frame to the thread so it doesn't freeze the video feed
            threading.Thread(target=send_alert_worker, args=(annotated_frame.copy(),), daemon=True).start()
    else:
        cv2.putText(annotated_frame, "STATUS: SCENE SAFE", (w//2 - 150, 80), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Enterprise Threat Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()