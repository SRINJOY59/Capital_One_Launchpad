from PIL import Image, UnidentifiedImageError
from transformers import ViTImageProcessor, ViTForImageClassification

image_processor = ViTImageProcessor.from_pretrained('wambugu71/crop_leaf_diseases_vit')
model = ViTForImageClassification.from_pretrained(
    'wambugu1738/crop_leaf_diseases_vit',
    ignore_mismatched_sizes=True
)

def detect_crop_disease(image_path: str):
    try:
        image = Image.open(image_path)
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        probs = logits.softmax(dim=-1).detach().cpu().numpy()[0]
        top_indices = probs.argsort()[-3:][::-1]
        top_diseases = [
            {
                "disease": model.config.id2label[idx],
                "probability": float(probs[idx])
            }
            for idx in top_indices
        ]
        return top_diseases
    except UnidentifiedImageError:
        return [{"error": "Invalid image file."}]
    except Exception as e:
        return [{"error": f"Error: {str(e)}"}]