# yeh bhi chal raha hai pass yolo hai 
import cv2
import numpy as np
import math
from ultralytics import YOLO
from collections import defaultdict, deque

print("Booting up the Kinematic Fight Detection Engine...")
model = YOLO('yolov8n-pose.pt') 

# Tracking dictionaries
# Stores the last few positions of wrists to calculate speed
wrist_history = defaultdict(lambda: deque(maxlen=3)) 
# Stores recent fight classifications to prevent flickering alarms
fight_buffer = deque(maxlen=10) 

# Thresholds (You will tune these for your presentation room!)
VELOCITY_THRESHOLD = 40 # Pixel distance moved per frame by a wrist
PROXIMITY_THRESHOLD = 200 # Pixel distance between two people's centers

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture('newfi11.avi')
print("\nSystem Armed. Press 'q' to quit.")

def calc_distance(p1, p2):
    """Calculates pixel distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model.track(frame, persist=True, classes=0, verbose=False)
    
    is_fighting = False
    annotated_frame = results[0].plot() # Draws the skeletons
    
    active_people = []

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        keypoints = results[0].keypoints.xy.cpu().numpy()
        boxes = results[0].boxes.xywh.cpu().numpy() # Get box centers for proximity

        # 1. Gather Data & Calculate Velocity for each person
        for i, track_id in enumerate(track_ids):
            kpts = keypoints[i]
            box = boxes[i]
            center_x, center_y = box[0], box[1]
            
            # Index 9: Left Wrist, Index 10: Right Wrist
            if len(kpts) > 10 and kpts[9][0] != 0 and kpts[10][0] != 0:
                l_wrist, r_wrist = kpts[9], kpts[10]
                
                # Get the person's history
                history = wrist_history[track_id]
                
                # Calculate speed if we have a previous frame
                max_speed = 0
                if len(history) > 0:
                    prev_l_wrist, prev_r_wrist = history[-1]
                    l_speed = calc_distance(l_wrist, prev_l_wrist)
                    r_speed = calc_distance(r_wrist, prev_r_wrist)
                    max_speed = max(l_speed, r_speed)
                
                # Save current position for the next frame
                history.append((l_wrist, r_wrist))
                
                # Store data for the proximity check
                active_people.append({
                    'id': track_id,
                    'center': (center_x, center_y),
                    'speed': max_speed
                })

        # 2. Check for Fights (Proximity + High Velocity)
        # We need at least 2 people in the frame to have a physical fight
        if len(active_people) >= 2:
            # Compare every person to every other person
            for i in range(len(active_people)):
                for j in range(i + 1, len(active_people)):
                    p1 = active_people[i]
                    p2 = active_people[j]
                    
                    # How close are they?
                    distance = calc_distance(p1['center'], p2['center'])
                    
                    # Are either of them throwing hands?
                    aggressive_movement = (p1['speed'] > VELOCITY_THRESHOLD) or (p2['speed'] > VELOCITY_THRESHOLD)
                    
                    # THE GOLDEN RULE: Close together AND moving fast
                    if distance < PROXIMITY_THRESHOLD and aggressive_movement:
                        is_fighting = True
                        break # Stop checking, we found a fight

    # 3. Temporal Smoothing (Don't trigger on a single glitchy frame)
    fight_buffer.append(1 if is_fighting else 0)
    
    # If 4 out of the last 10 frames detected a fight, trigger the alarm
    if sum(fight_buffer) >= 4:
        cv2.putText(annotated_frame, "🚨 KINEMATIC FIGHT DETECTED! 🚨", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.rectangle(annotated_frame, (0,0), (annotated_frame.shape[1], annotated_frame.shape[0]), (0,0,255), 10)
    else:
        cv2.putText(annotated_frame, "✅ SCENE SAFE", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    cv2.imshow("Production Grade Skeletal Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()