import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------------- Load Model and Video ---------------------- #

# Load fine-tuned YOLOv11 model for player detection
model = YOLO("LiatAI_assignment/best.pt")

# Load the video
video_path = "15sec_input_720p.mp4"
cap = cv2.VideoCapture(video_path)


width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


out = cv2.VideoWriter("player_reid_output.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# ---------------------- DeepSORT Tracker ---------------------- #

tracker = DeepSort(
    max_age=60,               # how long to keep 'lost' tracks
    n_init=2,                 # how many detections before confirming identity
    max_cosine_distance=0.3,  # stricter appearance matching
    nn_budget=100            # embedding buffer
)

# ---------------------- Process Video ---------------------- #

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0 and conf > 0.5:  # class 0 is player
            bbox = [x1, y1, x2 - x1, y2 - y1]  # format: x, y, w, h
            detections.append((bbox, conf, "player"))

    # Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Draw bounding box and player ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Player {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Write frame to output video
    out.write(frame)

    # Optional real-time display (comment if too slow)
    cv2.imshow("Player Re-ID", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break


cap.release()
out.release()
cv2.destroyAllWindows()
