from ultralytics import YOLO

def detect_pests(image_path: str, model_path: str = "../models/Pest_prediction/best.pt", imgsz: int = 640):
    model = YOLO(model_path)
    results = model.predict(image_path, imgsz=imgsz)
    detected_pests = set()
    if results and hasattr(results[0], "boxes"):
        names = results[0].names
        boxes = results[0].boxes
        if boxes is not None and hasattr(boxes, "cls"):
            class_indices = boxes.cls.cpu().numpy().astype(int)
            for idx in class_indices:
                detected_pests.add(names.get(idx, str(idx)))
    return list(detected_pests)

if __name__ == "__main__":
    pests = detect_pests("../Dataset/pest/test/bollworm/jpg_0.jpg")
    print("Detected pests:", pests)

"""
This response is the output of the Ultralytics YOLO object detection model after running inference on your pest image.

Key points:
- `results`: A list of Results objects, one per image processed.
- `boxes`: Contains the bounding boxes for detected objects (pests) in the image.
- `names`: Dictionary mapping class indices to pest names (e.g., 39: 'beet army worm', 86: 'Prodenia litura').
- The printed line shows:
  - The image path and its size (640x512).
  - Detected pests: "1 beet army worm, 1 Prodenia litura" (these are the pest classes detected in the image).
  - Inference speed details.
- The Results object contains:
  - `boxes`: Detected pest locations and class indices.
  - `names`: All possible pest classes the model can detect.
  - `orig_img`, `orig_shape`, `path`: Image data and metadata.
  - `probs`: (None here) would contain class probabilities for classification tasks.
  - `save_dir`: Directory where detection results are saved (with bounding boxes drawn).
  - `speed`: Timing for each step.

**Summary:**  
The model detected two pests in your image: "beet army worm" and "Prodenia litura".  
You can access bounding box coordinates and class indices via `results[0].boxes` and map them to pest names using `results[0].names`.
"""