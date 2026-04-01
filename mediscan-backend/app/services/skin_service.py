import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import timm
import cv2
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "app/models/best_convnext_advanced.pth"
DEVICE = "cpu"
NUM_CLASSES = 22

SKIN_CLASSES = [
    'Acne', 'Actinic_Keratosis', 'Benign_tumors', 'Bullous',
    'Candidiasis', 'DrugEruption', 'Eczema', 'Infestations_Bites',
    'Lichen', 'Lupus', 'Moles', 'Psoriasis', 'Rosacea',
    'Seborrh_Keratoses', 'SkinCancer', 'Sun_Sunlight_Damage',
    'Tinea', 'Unknown_Normal', 'Vascular_Tumors', 'Vasculitis',
    'Vitiligo', 'Warts'
]

CONFIDENCE_THRESHOLD = 0.6

# -----------------------------
# Model
# -----------------------------
model = timm.create_model(
    'convnext_base',
    pretrained=False,
    num_classes=NUM_CLASSES
)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=True)
model.eval()

# -----------------------------
# Preprocessing (ConvNeXt)
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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


def resize_with_padding(image_np, size=224):
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
# TTA Prediction
# -----------------------------
def predict_with_tta(image, model):
    images = [
        image,
        torch.flip(image, dims=[3])  # horizontal flip
    ]

    preds = []
    with torch.no_grad():
        for img in images:
            out = model(img)
            preds.append(F.softmax(out, dim=1))

    return torch.mean(torch.stack(preds), dim=0)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_skin(file):
    try:
        # Load image
        pil_image = Image.open(file.file).convert("RGB")
        image_np = np.array(pil_image)

        # -------- FILTERING PIPELINE --------

        # 1. Blur check
        if is_blurry(image_np):
            return {
                "disease": "Skin Disease",
                "prediction": "Invalid Image",
                "confidence": 0,
                "message": "Image is too blurry. Please upload a clearer image."
            }

        # 2. Enhance contrast
        image_np = enhance_contrast(image_np)

        # 3. Denoise
        image_np = denoise(image_np)

        # 4. Resize with padding
        image_np = resize_with_padding(image_np, 224)

        # Convert back to PIL
        image = Image.fromarray(image_np)

        # Apply transform
        image = transform(image).unsqueeze(0)

        # -------- MODEL PREDICTION --------
        probs = predict_with_tta(image, model)

        confidence, idx = torch.max(probs, dim=1)
        confidence = confidence.item()
        idx = idx.item()

        # -------- CONFIDENCE FILTER --------
        if confidence < CONFIDENCE_THRESHOLD or (
            confidence < 0.75 and SKIN_CLASSES[idx] == "Unknown_Normal"
        ):
            return {
                "disease": "Skin Disease",
                "prediction": "Uncertain",
                "confidence": round(confidence, 4),
                "message": "Low confidence. Please upload a clearer image or consult a dermatologist."
            }

        return {
            "disease": "Skin Disease",
            "prediction": SKIN_CLASSES[idx],
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {
            "disease": "Skin Disease",
            "prediction": "Error",
            "confidence": 0,
            "message": f"Error processing image: {str(e)}"
        }