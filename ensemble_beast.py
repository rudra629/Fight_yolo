# import cv2
# import math
# import torch
# import threading
# import time
# from PIL import Image
# from ultralytics import YOLO
# from transformers import CLIPProcessor, CLIPModel
# from collections import defaultdict, deque

# print("Initializing Tri-Factor Ensemble Engine...")

# # ==========================================
# # 1. LOAD THE MODELS (YOLO + CLIP)
# # ==========================================
# print("Loading YOLOv8 Kinematics...")
# yolo_model = YOLO('yolov8n-pose.pt')

# print("Loading OpenAI CLIP Semantics...")
# clip_id = "openai/clip-vit-base-patch32"
# clip_processor = CLIPProcessor.from_pretrained(clip_id)
# clip_model = CLIPModel.from_pretrained(clip_id)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# clip_model.to(device)
# clip_model.eval()

# # ==========================================
# # 2. GLOBAL VARIABLES & THREADING LOGIC
# # ==========================================
# latest_frame = None
# clip_is_fighting = False
# clip_confidence = 0.0

# # This function runs constantly in the background so it doesn't freeze your camera!
# def run_clip_in_background():
#     global latest_frame, clip_is_fighting, clip_confidence
    
#     candidate_labels = ["a violent physical fight between people", "normal everyday behavior, safe"]
    
#     while True:
#         if latest_frame is not None:
#             try:
#                 # Grab a copy of the current frame
#                 frame_to_analyze = latest_frame.copy()
#                 rgb_frame = cv2.cvtColor(frame_to_analyze, cv2.COLOR_BGR2RGB)
#                 pil_img = Image.fromarray(rgb_frame)
                
#                 # Ask CLIP what is happening
#                 with torch.no_grad():
#                     inputs = clip_processor(text=candidate_labels, images=pil_img, return_tensors="pt", padding=True).to(device)
#                     outputs = clip_model(**inputs)
#                     probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()
                    
#                     clip_confidence = probs[0] # Probability of "fight"
                    
#                     # CLIP must be over 60% sure it's a fight
#                     clip_is_fighting = clip_confidence > 0.60 
#             except Exception as e:
#                 pass
                
#         # Run this check 3 times a second
#         time.sleep(0.3)

# # Start the background thread
# threading.Thread(target=run_clip_in_background, daemon=True).start()

# # ==========================================
# # 3. MAIN YOLO ENGINE (Kinematics)
# # ==========================================
# wrist_history = defaultdict(lambda: deque(maxlen=3)) 
# fight_buffer = deque(maxlen=8) 

# # THRESHOLDS
# VELOCITY_THRESHOLD = 50 
# PROXIMITY_THRESHOLD = 200 

# print("\nSystem Armed. Show them what you built. Press 'q' to quit.")
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# def calc_distance(p1, p2):
#     return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

# while True:
#     ret, frame = cap.read()
#     if not ret: break
    
#     # Send the frame to the background CLIP thread
#     latest_frame = frame 
    
#     results = yolo_model.track(frame, persist=True, classes=0, verbose=False)
#     yolo_is_fighting = False
#     annotated_frame = results[0].plot() 
#     active_people = []

#     if results[0].boxes.id is not None:
#         track_ids = results[0].boxes.id.int().cpu().tolist()
#         keypoints = results[0].keypoints.xy.cpu().numpy()
#         boxes = results[0].boxes.xywh.cpu().numpy() 

#         for i, track_id in enumerate(track_ids):
#             kpts = keypoints[i]
#             box = boxes[i]
#             center_x, center_y = box[0], box[1]
            
#             if len(kpts) > 10 and kpts[9][0] != 0 and kpts[10][0] != 0:
#                 l_wrist, r_wrist = kpts[9], kpts[10]
#                 history = wrist_history[track_id]
                
#                 max_speed = 0
#                 if len(history) > 0:
#                     l_speed = calc_distance(l_wrist, history[-1][0])
#                     r_speed = calc_distance(r_wrist, history[-1][1])
#                     max_speed = max(l_speed, r_speed)
                
#                 history.append((l_wrist, r_wrist))
#                 active_people.append({'id': track_id, 'center': (center_x, center_y), 'speed': max_speed})

#         if len(active_people) >= 2:
#             for i in range(len(active_people)):
#                 for j in range(i + 1, len(active_people)):
#                     p1, p2 = active_people[i], active_people[j]
#                     distance = calc_distance(p1['center'], p2['center'])
#                     aggressive_movement = (p1['speed'] > VELOCITY_THRESHOLD) or (p2['speed'] > VELOCITY_THRESHOLD)
                    
#                     if distance < PROXIMITY_THRESHOLD and aggressive_movement:
#                         yolo_is_fighting = True
#                         break 

#     # ==========================================
#     # 4. THE MASTER "AND" GATE (Tri-Factor Check)
#     # ==========================================
#     # BOTH YOLO and CLIP must agree a fight is happening!
#     absolute_fight = yolo_is_fighting and clip_is_fighting
    
#     fight_buffer.append(1 if absolute_fight else 0)
    
#     # UI Display Logic
#     if sum(fight_buffer) >= 3:
#         cv2.putText(annotated_frame, "🚨 ABSOLUTE FIGHT DETECTED! 🚨", (20, 80), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
#         cv2.putText(annotated_frame, f"CLIP Confidence: {clip_confidence*100:.0f}%", (20, 130), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#         cv2.rectangle(annotated_frame, (0,0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0,0,255), 8)
#     else:
#         cv2.putText(annotated_frame, "✅ SCENE SAFE", (20, 80), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
#         # Show debug info so the judges know CLIP is working in the background
#         cv2.putText(annotated_frame, f"Semantic Context: {clip_confidence*100:.0f}%", (20, 130), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     cv2.imshow("Tri-Factor Ensemble Architecture", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()