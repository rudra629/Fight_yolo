import cv2
import math
import torch
import threading
import time
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict, deque

print("Initializing SOLO TEST Tri-Factor Engine with Live Logging...")

# ==========================================
# 1. LOAD THE MODELS
# ==========================================
print("Loading YOLOv8s Kinematics...")
yolo_model = YOLO('yolov8s-pose.pt')

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
                    clip_is_fighting = clip_confidence > 0.65 
            except Exception:
                pass
        time.sleep(0.3)

threading.Thread(target=run_clip_in_background, daemon=True).start()

# ==========================================
# 3. KINEMATIC ENGINE (YOLO)
# ==========================================
body_history = defaultdict(lambda: deque(maxlen=3)) 
fight_buffer = deque(maxlen=10) 

print("\nSystem Armed for SOLO TESTING. Press 'q' to quit.")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def calc_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    ret, frame = cap.read()
    if not ret: break
    
    latest_frame = frame 
    
    results = yolo_model.track(frame, persist=True, classes=0, conf=0.65, verbose=False)
    yolo_is_fighting = False
    annotated_frame = results[0].plot() 
    active_people = []

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xywh.cpu().numpy()

        for i, track_id in enumerate(track_ids):
            box = boxes[i]
            center_x, center_y, width, height = box[0], box[1], box[2], box[3]
            
            history = body_history[track_id]
            
            body_speed = 0
            if len(history) > 0:
                prev_center = history[-1]
                body_speed = calc_distance((center_x, center_y), prev_center)
            
            history.append((center_x, center_y))
            
            active_people.append({
                'id': track_id, 
                'center': (center_x, center_y), 
                'width': width,
                'speed': body_speed
            })

        # ==========================================
        # 4. SOLO COLLISION CHECK WITH LOGGING
        # ==========================================
        if len(active_people) >= 1:
            p1 = active_people[0] # Just grab the first person it sees (YOU)
            
            # GOD MODE MATH (5% of body width)
            dynamic_vel_p1 = p1['width'] * 0.05 
            
            # TERMINAL LOGGER 
            if p1['speed'] > 2.0:
                print(f"🎯 [GOD MATH] Your Width: {p1['width']:.0f}px | Speed Needed to Trigger: {dynamic_vel_p1:.0f} | Your Speed: {p1['speed']:.0f}")
            
            # Are YOU violently lunging/shifting?
            if p1['speed'] > dynamic_vel_p1:
                yolo_is_fighting = True

    # ==========================================
    # 5. THE MASTER GATE & SMOOTHING
    # ==========================================
    # Temporary Solo Override: Ignoring CLIP so you can see the red screen alone
    absolute_fight = yolo_is_fighting 
    fight_buffer.append(1 if absolute_fight else 0)
    
    # Trigger instantly on 1 frame for testing
    if sum(fight_buffer) >= 1:
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

    cv2.imshow("SOLO Tri-Factor Architecture", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()