import numpy as np
import tensorflow as tf
from PIL import Image
import io

# ===============================
# MODEL
# ===============================
MODEL_PATH = "app/models/malaria_128_FINAL.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# ⚠️ MUST MATCH TRAINING ORDER
CLASS_LABELS = ['Parasitized', 'Uninfected']   # check order in notebook

# ===============================
# Preprocessing (128×128)
# ===============================
def preprocess_image(image_bytes):             # ← changed parameter
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ===============================
# Prediction (updated)
# ===============================
def predict_malaria(image_bytes):              # ← changed parameter
    img = preprocess_image(image_bytes)        # ← pass bytes directly
    prob = float(model.predict(img, verbose=0)[0][0])
    print("DEBUG → raw sigmoid output:", prob)

    idx = 1 if prob >= 0.5 else 0
    prediction = CLASS_LABELS[idx]
    confidence = prob if idx == 1 else 1 - prob

    return {
        "disease": "Malaria",
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }