import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "app/models/brain_tumor_fixed.keras"
IMG_SIZE = 299

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']


def preprocess_image(file):
    image = Image.open(file.file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_brain_tumor(file):
    image = preprocess_image(file)
    preds = model.predict(image)
    idx = int(np.argmax(preds))
    return {
        "disease": "Brain Tumor",
        "prediction": CLASS_LABELS[idx],
        "confidence": float(np.max(preds))
    }
