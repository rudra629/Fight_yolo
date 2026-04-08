import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

print("Downloading YOLOv8 Pose Model... (Just takes a few seconds)")
# Load the lightweight pose tracking model
model = YOLO('yolov8n-pose.pt') 

# Store the history of wrist positions for calculating speed
# Dictionary to store tracking IDs and their recent keypoint coordinates
track_history = defaultdict(lambda: [])

# Boot up camera (DirectShow for Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("\nSkeletal Tracking Online! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Run YOLOv8 Pose inference with built-in tracking!
    # persist=True assigns a unique ID to every person it sees
    results = model.track(frame, persist=True, classes=0, verbose=False)

    if results[0].boxes.id is not None:
        # Extract tracking IDs and keypoints (skeletons)
        track_ids = results[0].boxes.id.int().cpu().tolist()
        keypoints = results[0].keypoints.xy.cpu().numpy() # x, y coordinates of all joints

        # Draw the skeletons on the frame (The "Wow" Factor)
        annotated_frame = results[0].plot()

        # Let's do some basic math on the wrists to prove it works
        for track_id, kpts in zip(track_ids, keypoints):
            # In YOLOv8, index 9 is Left Wrist, index 10 is Right Wrist
            # Check if wrists are actually visible (not [0,0])
            if len(kpts) > 10 and kpts[9][0] != 0 and kpts[10][0] != 0:
                left_wrist = kpts[9]
                right_wrist = kpts[10]
                
                # Store the wrist positions for this specific person
                history = track_history[track_id]
                history.append((left_wrist, right_wrist))
                
                # Keep only the last 5 frames of memory
                if len(history) > 5:
                    history.pop(0)

                # --- THIS IS WHERE WE WILL ADD THE VELOCITY MATH NEXT ---
                # Right now, we are just proving we can track unique people!
                cv2.putText(annotated_frame, f"ID: {track_id} Tracking", (int(left_wrist[0]), int(left_wrist[1])-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    else:
        # If no people are found, just show the normal frame
        annotated_frame = frame

    cv2.imshow("Production Grade Skeletal Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()