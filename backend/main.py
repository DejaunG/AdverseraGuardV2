from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import io
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import base64
import numpy as np
from torchvision import transforms, models
from adversarial_methods import generate_adversarial
import logging

app = FastAPI()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained model
num_classes = 4  # Adjust according to the dataset
class_names = ['Edible', 'Fresh', 'Non-fresh', 'Poisonous']
num_classes = len(class_names)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
model.load_state_dict(torch.load('adversera_model.pth', map_location=device))
model.eval()
model.to(device)

# Define your data transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def classify_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)
    return class_names[predicted.item()]


@app.post("/process")
async def process_image(
        image: UploadFile = File(...),
        attack_method: str = Form(...)
):
    try:
        logger.info("Received image and attack method: %s", attack_method)
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        logger.info("Image preprocessed")

        # Classify original image
        original_result = classify_image(model, input_tensor)
        logger.info("Original image classified as %s", original_result)

        # Generate adversarial image
        adv_image = generate_adversarial(model, input_tensor, attack_method)
        logger.info("Adversarial image generated")

        # Classify adversarial image
        adv_result = classify_image(model, adv_image)
        logger.info("Adversarial image classified as %s", adv_result)

        # Convert tensor to PIL Image
        adv_img_pil = transforms.ToPILImage()(adv_image.squeeze().cpu())

        # Encode image to base64
        buffered = io.BytesIO()
        adv_img_pil.save(buffered, format="JPEG")
        adv_image_str = base64.b64encode(buffered.getvalue()).decode()
        logger.info("Adversarial image encoded")

        return JSONResponse(content={
            'adv_image': adv_image_str,
            'original_result': original_result,
            'adv_result': adv_result
        })
    except Exception as e:
        logger.exception("Error processing image: %s", str(e))
        return JSONResponse(content={'error': str(e)}, status_code=500)
