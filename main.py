import io
from pathlib import Path

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import models, transforms
import uvicorn


# --------------------------------------------------
# 1. MODEL
# --------------------------------------------------

class SiameseResNet50(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        resnet = models.resnet50(weights=None)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = resnet.fc.in_features * 3

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x):
        x = self.backbone(x)
        return torch.flatten(x, 1)

    def forward(self, img_pre, img_post):
        f_pre = self.extract_features(img_pre)
        f_post = self.extract_features(img_post)
        f_diff = torch.abs(f_pre - f_post)

        features = torch.cat((f_pre, f_post, f_diff), dim=1)
        return self.classifier(features)


# --------------------------------------------------
# 2. SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "siamese_resnet50_xbd_4class.joblib"

LABEL_NAMES = [
    "No Damage (0)",
    "Minor (1)",
    "Major/Destroyed (2)",
    "Un-classified (3)"
]

device = torch.device("cpu")


# --------------------------------------------------
# 3. LOAD MODEL
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = SiameseResNet50(num_classes=4)

    # The checkpoint is a joblib file containing PyTorch tensors.
    # Force CUDA-saved tensors onto the CPU while loading.
    original_loader = torch.storage._load_from_bytes

    try:
        torch.storage._load_from_bytes = lambda data: torch.load(
            io.BytesIO(data),
            map_location="cpu",
            weights_only=False
        )
        state_dict = joblib.load(MODEL_PATH)
    finally:
        torch.storage._load_from_bytes = original_loader

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# 4. IMAGE PREPROCESSING
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def process_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image)
        return image.unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


# --------------------------------------------------
# 5. FASTAPI
# --------------------------------------------------

app = FastAPI(
    title="xBD Damage Classification API",
    description="Local Siamese ResNet-50 damage classification API"
)


@app.get("/")
def home():
    return {
        "status": "running",
        "model": "Siamese ResNet-50",
        "device": "cpu",
        "classes": LABEL_NAMES
    }


@app.post("/predict")
async def predict_damage(
    file_pre: UploadFile = File(...),
    file_post: UploadFile = File(...)
):
    pre_bytes = await file_pre.read()
    post_bytes = await file_post.read()

    if not pre_bytes or not post_bytes:
        raise HTTPException(
            status_code=400,
            detail="Please upload both pre-event and post-event images."
        )

    pre_image = process_image(pre_bytes)
    post_image = process_image(post_bytes)

    with torch.no_grad():
        logits = model(pre_image, post_image)
        probabilities = F.softmax(logits, dim=1)[0]

    predicted_class = int(torch.argmax(probabilities).item())

    class_probabilities = {
        LABEL_NAMES[i]: round(float(probabilities[i]), 4)
        for i in range(4)
    }

    return {
        "prediction_id": predicted_class,
        "prediction_label": LABEL_NAMES[predicted_class],
        "confidence": round(float(probabilities[predicted_class]), 4),
        "class_probabilities": class_probabilities
    }


# --------------------------------------------------
# 6. RUN LOCALLY
# --------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
