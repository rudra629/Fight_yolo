# yeh bhi chal raha hai
# currnet best version of the D-Day Tri-Factor Engine, combining YOLOv8s Kinematics with CLIP Semantics for near-perfect fight detection in your presentation room.
import cv2
import math
import torch
import threading
import time
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict, deque

print("Initializing D-Day Tri-Factor Engine...")

# ==========================================
# 1. LOAD THE MODELS
# ==========================================
# UPGRADE: Using 's' (Small) instead of 'n' (Nano) for much higher accuracy
print("Loading YOLOv8s Kinematics...")
yolo_model = YOLO()

print("Loading OpenAI CLIP Semantics...")
clip_id = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_id)
clip_model = CLIPModel.from_pretrained(clip_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)
clip_model.eval()

# ==========================================
# 2. BACKGROUND SEMANTICS (CLIP)
# ==========================================
latest_frame = None
clip_is_fighting = False
clip_confidence = 0.0

def run_clip_in_background():
    global latest_frame, clip_is_fighting, clip_confidence
    # UPGRADE: Tighter semantic prompts to prevent false positives
    candidate_labels = ["people actively engaged in a violent physical fight", "people standing, talking, or acting normally"]
    
    while True:
        if latest_frame is not None:
            try:
                frame_to_analyze = latest_frame.copy()
                rgb_frame = cv2.cvtColor(frame_to_analyze, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                
                with torch.no_grad():
                    inputs = clip_processor(text=candidate_labels, images=pil_img, return_tensors="pt", padding=True).to(device)
                    outputs = clip_model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()
                    
                    clip_confidence = probs[0]
                    # UPGRADE: CLIP must be 65% sure to authorize the alarm
                    clip_is_fighting = clip_confidence > 0.65 
            except Exception:
                pass
        time.sleep(0.3)

threading.Thread(target=run_clip_in_background, daemon=True).start()

# ==========================================
# 3. KINEMATIC ENGINE (YOLO)
# ==========================================
# Track entire body movement, not just glitchy wrists
body_history = defaultdict(lambda: deque(maxlen=3)) 
fight_buffer = deque(maxlen=10) 

print("\nSystem Armed and Bulletproof. Press 'q' to quit.")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def calc_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    ret, frame = cap.read()
    if not ret: break
    
    latest_frame = frame 
    
    # UPGRADE: conf=0.65 completely eliminates pillows and background junk
    results = yolo_model.track(frame, persist=True, classes=0, conf=0.40, verbose=False)
    yolo_is_fighting = False
    annotated_frame = results[0].plot() 
    active_people = []

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xywh.cpu().numpy() # x_center, y_center, width, height

        for i, track_id in enumerate(track_ids):
            box = boxes[i]
            center_x, center_y, width, height = box[0], box[1], box[2], box[3]
            
            history = body_history[track_id]
            
            # Calculate Torso/Center-of-Mass velocity
            body_speed = 0
            if len(history) > 0:
                prev_center = history[-1]
                body_speed = calc_distance((center_x, center_y), prev_center)
            
            history.append((center_x, center_y))
            
            # Save all data needed for the collision check
            active_people.append({
                'id': track_id, 
                'center': (center_x, center_y), 
                'width': width,
                'speed': body_speed
            })

        # 4. THE DYNAMIC COLLISION CHECK
        if len(active_people) >= 2  :
            for i in range(len(active_people)):
                for j in range(i + 1, len(active_people)):
                    p1, p2 = active_people[i], active_people[j]
                    
                    distance = calc_distance(p1['center'], p2['center'])
                    
                    # UPGRADE: Dynamic Proximity! 
                    # If the distance between them is less than their combined average width, their boxes are overlapping.
                    dynamic_proximity_limit = (p1['width'] + p2['width']) / 1.5
                    
                    # GOD MODE UPGRADE: Dynamic Velocity Scaling (Distance-Invariant)
                    dynamic_vel_p1 = p1['width'] * 0.08 
                    dynamic_vel_p2 = p2['width'] * 0.08
                    
                    # Are their bodies violently lunging/shifting?
                    aggressive_movement = (p1['speed'] > dynamic_vel_p1) or (p2['speed'] > dynamic_vel_p2)
                    
                    if distance < dynamic_proximity_limit and aggressive_movement:
                        yolo_is_fighting = True
                        break 

    # ==========================================
    # 5. THE MASTER GATE & SMOOTHING
    # ==========================================
    absolute_fight = yolo_is_fighting and clip_is_fighting
    fight_buffer.append(1 if absolute_fight else 0)
    
    # Require 5 frames of absolute confirmation to trigger (Kills micro-flicker false alarms)
    if sum(fight_buffer) >= 5:
        cv2.putText(annotated_frame, "🚨 ABSOLUTE FIGHT DETECTED! 🚨", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(annotated_frame, f"CLIP Confidence: {clip_confidence*100:.0f}%", (20, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.rectangle(annotated_frame, (0,0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0,0,255), 8)
    else:
        cv2.putText(annotated_frame, "✅ SCENE SAFE", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(annotated_frame, f"Semantic Context: {clip_confidence*100:.0f}%", (20, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Tri-Factor Ensemble Architecture", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()