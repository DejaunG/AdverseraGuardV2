import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from adversarial_methods import generate_adversarial_example
import logging

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

# Load the pre-trained model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Define your custom classes
custom_classes = {
    'fish_eye': ['fresh', 'non-fresh'],
    'mushroom': ['poisonous', 'non-poisonous']
}

def get_classification(pred_index, image_type):
    if image_type in custom_classes:
        return custom_classes[image_type][pred_index % len(custom_classes[image_type])]
    else:
        return f"Class {pred_index}"

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
    logger.info(f"Received request: method={method}, epsilon={epsilon}, stealth_mode={stealth_mode}, image_type={image_type}")
    logger.debug(f"All parameters: {locals()}")
    try:
        contents = await file.read()
        logger.info(f"File contents size: {len(contents)} bytes")
        image = Image.open(io.BytesIO(contents))
        logger.info(f"Image opened successfully: {image.format}, {image.size}, {image.mode}")

        # Preprocess the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        # Generate adversarial example
        with torch.no_grad():
            output = model(input_batch)
        original_pred = output.argmax().item()

        adversarial_image = generate_adversarial_example(
            model, input_batch, torch.tensor([original_pred]), method,
            epsilon=epsilon, alpha=alpha, num_iter=num_iter,
            num_classes=num_classes, overshoot=overshoot, max_iter=max_iter,
            pixels=pixels, pop_size=pop_size, delta=delta,
            max_iter_uni=max_iter_uni, max_iter_df=max_iter_df,
            stealth_mode=stealth_mode
        )

        # Get prediction for adversarial image
        with torch.no_grad():
            adv_output = model(adversarial_image)
        adv_pred = adv_output.argmax().item()

        # Convert tensor to PIL Image for saving
        to_pil = transforms.ToPILImage()
        adv_image_pil = to_pil(adversarial_image.squeeze(0))

        # Save adversarial image to bytes
        img_byte_arr = io.BytesIO()
        adv_image_pil.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Encode image to base64
        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        # Get classifications
        original_class = get_classification(original_pred, image_type)
        adversarial_class = get_classification(adv_pred, image_type)

        logger.info(f"Successfully generated adversarial image: original_pred={original_pred}, adv_pred={adv_pred}")
        logger.info(f"Classifications: original={original_class}, adversarial={adversarial_class}")

        return JSONResponse({
            "original_prediction": original_class,
            "adversarial_prediction": adversarial_class,
            "adversarial_image": img_base64
        })
    except Exception as e:
        logger.exception(f"Error generating adversarial image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)