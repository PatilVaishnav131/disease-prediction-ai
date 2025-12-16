import numpy as np
import tensorflow as tf
from PIL import Image
import io

# ===============================
# ===============================
MODEL_PATH = "app/models/malaria_128_FINAL.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ⚠️ MUST MATCH TRAINING ORDER
# Check this ONCE in notebook:
# print(train_generator.class_indices)
CLASS_LABELS = ['Parasitized', 'Uninfected']  
# CHANGE ORDER if needed

# ===============================
# Preprocessing (128×128)
# ===============================
def preprocess_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# ===============================
# Prediction
# ===============================
def predict_malaria(file):
    img = preprocess_image(file.file.read())

    prob = float(model.predict(img, verbose=0)[0][0])
    print("DEBUG → raw sigmoid output:", prob)

    # Binary sigmoid → class index
    idx = 1 if prob >= 0.5 else 0

    prediction = CLASS_LABELS[idx]
    confidence = prob if idx == 1 else 1 - prob

    return {
        "disease": "Malaria",
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }
