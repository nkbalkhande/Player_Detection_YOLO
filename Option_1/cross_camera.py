# Cross-Camera Player Mapping with YOLOv11 + DeepSORT + CLIP

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import clip

# ---------------------------- Setup ---------------------------- #
yolo_model_path = "LiatAI_assignment/best.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def extract_embedding(img):
    image = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

# ---------------------------- Detection + Tracking ---------------------------- #


def process_video(video_path, source_name):
    model = YOLO(yolo_model_path)
    tracker = DeepSort(max_age=60, n_init=2,
                       max_cosine_distance=0.3, nn_budget=100)

    cap = cv2.VideoCapture(video_path)
    embeddings_dict = {}

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

            if cls == 0 and conf > 0.5:
                bbox = [x1, y1, x2 - x1, y2 - y1]
                detections.append((bbox, conf, "player"))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            emb = extract_embedding(crop)
            embeddings_dict[track_id] = emb

    cap.release()
    return embeddings_dict


# ---------------------------- Run on Both Videos ---------------------------- #
print("Processing broadcast video...")
broadcast_embeddings = process_video("broadcast.mp4", "broadcast")
print("Processing tacticam video...")
tacticam_embeddings = process_video("tacticam.mp4", "tacticam")

# ---------------------------- Match IDs ---------------------------- #
print("Matching players across cameras...")
b_keys = list(broadcast_embeddings.keys())
t_keys = list(tacticam_embeddings.keys())

b_matrix = np.array([broadcast_embeddings[k] for k in b_keys])
t_matrix = np.array([tacticam_embeddings[k] for k in t_keys])

similarity_matrix = cosine_similarity(t_matrix, b_matrix)

mapping = {}
for i, t_id in enumerate(t_keys):
    best_match_index = np.argmax(similarity_matrix[i])
    b_id = b_keys[best_match_index]
    mapping[t_id] = b_id

# ---------------------------- Save Mapping ---------------------------- #
with open("player_id_mapping.csv", "w") as f:
    f.write("tacticam_id,broadcast_id\n")
    for t_id, b_id in mapping.items():
        f.write(f"{t_id},{b_id}\n")

print("Mapping complete. Saved to player_id_mapping.csv")
