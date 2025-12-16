
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import timm

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
# Rebuild EXACT training model
# -----------------------------
model = timm.create_model(
    'convnext_base',       # ✅ EXACT match
    pretrained=False,      # weights come from state_dict
    num_classes=NUM_CLASSES
)

# -----------------------------
# Load trained weights
# -----------------------------
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=True)
model.eval()

# -----------------------------
# ConvNeXt preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Prediction
# -----------------------------
def predict_skin(file):
    image = Image.open(file.file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    confidence, idx = torch.max(probs, dim=1)
    confidence = confidence.item()
    idx = idx.item()

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "disease": "Skin Disease",
            "prediction": "Uncertain",
            "confidence": round(confidence, 4),
            "message": "Low confidence. Please consult a dermatologist."
        }

    return {
        "disease": "Skin Disease",
        "prediction": SKIN_CLASSES[idx],
        "confidence": round(confidence, 4)
    }
