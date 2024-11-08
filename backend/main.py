import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
import io
import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b3
import urllib.parse
from fastapi.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up paths
BASE_DIR = Path(__file__).resolve().parent
GALLERY_DIR = BASE_DIR / "gallery"
MODEL_PATH = BASE_DIR / "optimized_adversera_model.pth"

# Create gallery directories if they don't exist
categories = ['fresh', 'non-fresh', 'edible', 'poisonous']
for category in categories:
    (GALLERY_DIR / category).mkdir(parents=True, exist_ok=True)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
classification_model = create_classification_model()
classification_model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
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

# Add denormalization transform
denormalize = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])

# Add normalization transform
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

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


@app.get("/gallery/{category}")
async def get_gallery_images(category: str):
    """Get list of images from a specific category"""
    logger.info(f"Accessing gallery category: {category}")

    try:
        if category not in categories:
            logger.error(f"Invalid category requested: {category}")
            raise HTTPException(status_code=404, detail=f"Category {category} not found")

        category_path = GALLERY_DIR / category
        if not category_path.exists():
            logger.error(f"Gallery folder not found: {category_path}")
            raise HTTPException(status_code=404, detail=f"Gallery folder {category} not found")

        # Get all image files
        valid_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        images = []

        logger.debug(f"Scanning directory: {category_path}")
        for file in category_path.iterdir():
            if file.is_file() and file.suffix.lower() in valid_extensions:
                # Just use the filename, not the full path
                images.append(file.name)
                logger.debug(f"Found image: {file.name}")

        logger.info(f"Found {len(images)} images in category {category}")
        return {"images": sorted(images)}

    except Exception as e:
        logger.exception(f"Error in get_gallery_images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gallery-image/{category}/{filename:path}")
async def get_gallery_image(category: str, filename: str):
    """Serve an image from the gallery"""
    logger.info(f"Requesting image: {category}/{filename}")

    try:
        # Decode the URL-encoded filename
        decoded_filename = urllib.parse.unquote(filename)
        image_path = GALLERY_DIR / category / decoded_filename

        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            raise HTTPException(status_code=404, detail="Image not found")

        # Security check
        try:
            image_path.relative_to(GALLERY_DIR)
        except ValueError:
            logger.error(f"Invalid image path: {image_path}")
            raise HTTPException(status_code=400, detail="Invalid image path")

        logger.debug(f"Serving image: {image_path}")

        # Read the image and return it as a response
        with open(image_path, "rb") as f:
            image_data = f.read()

        # Determine content type
        content_type = "image/jpeg"
        if filename.lower().endswith('.png'):
            content_type = "image/png"

        return Response(content=image_data, media_type=content_type)

    except Exception as e:
        logger.exception(f"Error serving gallery image {category}/{filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_image_type")
async def detect_image_type_endpoint(file: UploadFile = File(...)):
    """Detect image type endpoint with improved error handling"""
    try:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Please upload a valid image file."
            )

        if image.mode != 'RGB':
            image = image.convert('RGB')

        detected_type = detect_image_type(image)
        return JSONResponse({"image_type": detected_type})
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Error detecting image type: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the image."
        )


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
        try:
            image = Image.open(io.BytesIO(contents))
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Please upload a valid image file."
            )

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if image_type == 'detect' or image_type == 'auto':
            image_type = detect_image_type(image)
            logger.info(f"Detected image type: {image_type}")

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        original_class = get_classification(input_batch, image_type)
        logger.info(f"Original classification: {original_class}")

        try:
            if stealth_mode:
                logger.info("Using stealth mode with normalized perturbations")
                from adversarial_methods import generate_adversarial_example
                adversarial_image = generate_adversarial_example(
                    adversarial_model, input_batch, torch.tensor([0]).to(device), method,
                    epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                    num_classes=num_classes, overshoot=overshoot, max_iter=max_iter,
                    pixels=pixels, pop_size=pop_size, delta=delta,
                    max_iter_uni=max_iter_uni, max_iter_df=max_iter_df,
                    stealth_mode=stealth_mode
                )
                adversarial_class = get_classification(adversarial_image, image_type)
                display_image = denormalize(adversarial_image)
            else:
                logger.info("Using regular mode with denormalized perturbations")
                input_denorm = denormalize(input_batch)
                from adversarial_methods import generate_adversarial_example
                adversarial_image = generate_adversarial_example(
                    adversarial_model, input_denorm, torch.tensor([0]).to(device), method,
                    epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                    num_classes=num_classes, overshoot=overshoot, max_iter=max_iter,
                    pixels=pixels, pop_size=pop_size, delta=delta,
                    max_iter_uni=max_iter_uni, max_iter_df=max_iter_df,
                    stealth_mode=stealth_mode
                )
                adversarial_image = torch.clamp(adversarial_image, 0, 1)
                adv_normalized = normalize(adversarial_image[0]).unsqueeze(0)
                adversarial_class = get_classification(adv_normalized, image_type)
                display_image = adversarial_image

            to_pil = transforms.ToPILImage()
            adv_image_pil = to_pil(torch.clamp(display_image.squeeze(0).cpu(), 0, 1))

            img_byte_arr = io.BytesIO()
            adv_image_pil.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

            logger.info(f"Classifications: original={original_class}, adversarial={adversarial_class}")

            return JSONResponse({
                "original_prediction": f"{original_class} {image_type}",
                "adversarial_prediction": f"{adversarial_class} {image_type}",
                "adversarial_image": img_base64
            })

        except Exception as e:
            logger.exception(f"Error during adversarial generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

    except Exception as e:
        logger.exception(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Log initial configuration
    logger.info(f"Starting server with gallery directory: {GALLERY_DIR}")
    logger.info("Available categories: " + ", ".join(categories))

    # Log number of images in each category
    for category in categories:
        category_path = GALLERY_DIR / category
        num_images = len([f for f in category_path.glob("*") if f.is_file()])
        logger.info(f"Category {category}: {num_images} images")

    uvicorn.run(app, host="0.0.0.0", port=8000)