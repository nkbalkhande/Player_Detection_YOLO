# Player Re-Identification with YOLOv11, DeepSORT, and CLIP

This project performs **player detection, tracking, and cross-camera identity mapping** using a fine-tuned YOLOv11 model, DeepSORT tracker, and CLIP embeddings. It contains two modes:

- **Option 1**: Cross-Camera Player Mapping (broadcast vs tacticam video)
- **Option 2**: Single Video Player Re-identification with visualization

---

## Requirements

Install the following libraries:

```bash
pip install opencv-python torch torchvision numpy sklearn clip-by-openai
pip install ultralytics
pip install deep_sort_realtime

