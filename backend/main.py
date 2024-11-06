import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b3
import torch.nn as nn
from adversarial_methods import generate_adversarial_example
import logging
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load both models
def create_classification_model():
    model = efficientnet_b3(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 4)
    )
    return model


# Create and load models
model_path = os.path.join(os.path.dirname(__file__), 'optimized_adversera_model.pth')
classification_model = create_classification_model()
classification_model.load_state_dict(torch.load(model_path))
classification_model.eval()
classification_model = classification_model.to(device)

adversarial_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
adversarial_model.eval()
adversarial_model = adversarial_model.to(device)

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class names and mappings
class_names = ['fresh_fish_eye', 'non_fresh_fish_eye', 'poisonous_mushroom', 'non_poisonous_mushroom']


def get_classification(image_tensor, image_type):
    """Get classification from our custom model"""
    with torch.no_grad():
        output = classification_model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()
        class_name = class_names[prediction]

        if 'fish_eye' in class_name:
            return 'fresh' if 'fresh_' in class_name else 'non-fresh'
        else:
            return 'poisonous' if 'poisonous_' in class_name else 'non-poisonous'


def detect_image_type(image):
    """Detect image type using the custom trained model"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = classification_model(input_batch)
        prediction = torch.argmax(output, dim=1).item()
        predicted_class = class_names[prediction]

    if 'fish_eye' in predicted_class:
        return 'fish_eye'
    else:
        return 'mushroom'


@app.post("/detect_image_type")
async def detect_image_type_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        detected_type = detect_image_type(image)
        return JSONResponse({"image_type": detected_type})
    except Exception as e:
        logger.exception(f"Error detecting image type: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_adversarial")
async def generate_adversarial(
        file: UploadFile = File(...),
        method: str = Form(...),
        epsilon: float = Form(...),
        stealth_mode: bool = Form(...),
        image_type: str = Form(...),
        alpha: float = Form(0.01),
        num_iter: int = Form(40),
        num_classes: int = Form(1000),
        overshoot: float = Form(0.02),
        max_iter: int = Form(50),
        pixels: int = Form(1),
        pop_size: int = Form(400),
        delta: float = Form(0.2),
        max_iter_uni: int = Form(50),
        max_iter_df: int = Form(100)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if image_type == 'detect' or image_type == 'auto':
            image_type = detect_image_type(image)
            logger.info(f"Detected image type: {image_type}")

        # Preprocess the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Get original classification from our model
        original_class = get_classification(input_batch, image_type)

        # Generate adversarial example
        adversarial_image = generate_adversarial_example(
            adversarial_model, input_batch, torch.tensor([0]).to(device), method,
            epsilon=epsilon, alpha=alpha, num_iter=num_iter,
            num_classes=num_classes, overshoot=overshoot, max_iter=max_iter,
            pixels=pixels, pop_size=pop_size, delta=delta,
            max_iter_uni=max_iter_uni, max_iter_df=max_iter_df,
            stealth_mode=stealth_mode
        )

        # Get classification for adversarial image from our model
        adversarial_class = get_classification(adversarial_image, image_type)

        # Convert tensor to PIL Image
        to_pil = transforms.ToPILImage()
        adv_image_pil = to_pil(adversarial_image.squeeze(0))

        # Save to bytes
        img_byte_arr = io.BytesIO()
        adv_image_pil.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Encode to base64
        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        logger.info(f"Classifications: original={original_class}, adversarial={adversarial_class}")

        return JSONResponse({
            "original_prediction": f"{original_class} {image_type}",
            "adversarial_prediction": f"{adversarial_class} {image_type}",
            "adversarial_image": img_base64
        })

    except Exception as e:
        logger.exception(f"Error generating adversarial image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)