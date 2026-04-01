import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

MODEL_PATH = "app/models/brain_tumor_fixed.keras"
IMG_SIZE = 299

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']


# -----------------------------
# Filtering Functions
# -----------------------------
def is_blurry(image_np, threshold=100):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def enhance_contrast(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def denoise(image_np):
    return cv2.GaussianBlur(image_np, (5, 5), 0)


def resize_with_padding(image_np, size=299):
    h, w, _ = image_np.shape
    scale = size / max(h, w)

    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image_np, (new_w, new_h))

    pad_h = size - new_h
    pad_w = size - new_w

    padded = cv2.copyMakeBorder(
        resized,
        pad_h // 2, pad_h - pad_h // 2,
        pad_w // 2, pad_w - pad_w // 2,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return padded


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_image(file):
    # Load image
    pil_image = Image.open(file.file).convert("RGB")
    image_np = np.array(pil_image)

    # -------- FILTERING --------

    # Blur handling (do not reject)
    if is_blurry(image_np):
        image_np = enhance_contrast(image_np)

    # Contrast enhancement
    image_np = enhance_contrast(image_np)

    # Denoising
    image_np = denoise(image_np)

    # Resize safely
    image_np = resize_with_padding(image_np, IMG_SIZE)

    # Normalize
    image_np = image_np / 255.0
    image_np = np.expand_dims(image_np, axis=0)

    return image_np


# -----------------------------
# Prediction
# -----------------------------
def predict_brain_tumor(file):
    try:
        image = preprocess_image(file)

        preds = model.predict(image)
        confidence = float(np.max(preds))
        idx = int(np.argmax(preds))

        return {
            "disease": "Brain Tumor",
            "prediction": CLASS_LABELS[idx],
            "confidence": round(confidence, 4),
            "warning": "Low confidence prediction" if confidence < 0.6 else ""
        }

    except Exception as e:
        return {
            "disease": "Brain Tumor",
            "prediction": "Error",
            "confidence": 0,
            "message": f"Error processing image: {str(e)}"
        }