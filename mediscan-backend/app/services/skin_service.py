import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import timm
import cv2
import numpy as np
import requests
from io import BytesIO

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
# Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Helper Functions
# -----------------------------
def load_image_from_url(url):
    """Fetch an image from a URL and return a PIL Image."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def is_blurry(image_np, threshold=100):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def is_low_contrast(image_np, threshold=0.2):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    return np.std(gray) < threshold * 255

def enhance_contrast(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

def resize_and_crop(image_np, size=224):
    """
    Resize so the shorter side becomes `size`, then center crop to `size`×`size`.
    Includes fallbacks for pathological dimensions.
    """
    h, w = image_np.shape[:2]
    if h < 1 or w < 1:
        raise ValueError(f"Invalid image dimensions: {h}x{w}")
    
    scale = size / min(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Ensure at least size in both dimensions
    if new_h < size or new_w < size:
        # Fallback: direct resize
        resized = cv2.resize(image_np, (size, size), interpolation=cv2.INTER_LINEAR)
        return resized
    
    resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    start_h = (new_h - size) // 2
    start_w = (new_w - size) // 2
    cropped = resized[start_h:start_h+size, start_w:start_w+size]
    
    # Final safety check
    if cropped.shape[0] != size or cropped.shape[1] != size:
        cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
    return cropped

# -----------------------------
# TTA
# -----------------------------
def predict_with_tta(image, model):
    images = [image, torch.flip(image, dims=[3])]
    preds = []
    with torch.no_grad():
        for img in images:
            out = model(img)
            preds.append(F.softmax(out, dim=1))
    return torch.mean(torch.stack(preds), dim=0)

# -----------------------------
# Main Prediction Function
# -----------------------------
def predict_skin(image_data, from_url=False):
    """
    image_data: bytes (file upload) or str (URL if from_url=True)
    """
    try:
        # ---------- Load image ----------
        if from_url:
            pil_image = load_image_from_url(image_data)
        else:
            # Accept both bytes and file-like objects
            if isinstance(image_data, bytes):
                pil_image = Image.open(BytesIO(image_data)).convert("RGB")
            else:
                # For backward compatibility with file-like objects
                pil_image = Image.open(image_data).convert("RGB")
        
        image_np = np.array(pil_image)
        
        # ---------- Fix channel issues ----------
        if len(image_np.shape) == 2:  # grayscale
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 1:  # single channel
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # ---------- Quality-based enhancements ----------
        if is_blurry(image_np) or is_low_contrast(image_np):
            image_np = enhance_contrast(image_np)
        
        # ---------- Resize and crop to 224x224 ----------
        image_np = resize_and_crop(image_np, 224)
        
        # ---------- Transform to tensor ----------
        image = Image.fromarray(image_np)
        image_tensor = transform(image).unsqueeze(0)  # (1,3,224,224)
        
        # ---------- Final shape guard ----------
        if image_tensor.shape[2] != 224 or image_tensor.shape[3] != 224:
            image_tensor = F.interpolate(
                image_tensor, size=(224, 224), 
                mode='bilinear', align_corners=False
            )
        
        # ---------- Prediction ----------
        probs = predict_with_tta(image_tensor, model)[0]
        
        confidence, idx = torch.max(probs, dim=0)
        confidence = confidence.item()
        idx = idx.item()
        
        top3_probs, top3_idx = torch.topk(probs, 3)
        top_predictions = [
            {"class": SKIN_CLASSES[top3_idx[i].item()],
             "confidence": round(top3_probs[i].item(), 4)}
            for i in range(3)
        ]
        
        return {
            "disease": "Skin Disease",
            "prediction": SKIN_CLASSES[idx],
            "confidence": round(confidence, 4),
            "top_predictions": top_predictions,
            "warning": "Low confidence prediction" if confidence < 0.6 else ""
        }
    
    except Exception as e:
        return {
            "disease": "Skin Disease",
            "prediction": "Error",
            "confidence": 0,
            "message": f"Error processing image: {str(e)}"
        }