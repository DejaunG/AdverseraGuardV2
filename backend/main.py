import base64
import os
import time
from typing import List, Dict, Any, Optional
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from adversarial_methods import generate_adversarial_example, fgsm_attack, pgd_attack, deepfool_attack, one_pixel_attack
import logging
import numpy as np

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

# Load the pre-trained model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Enable cuDNN benchmarking for improved GPU performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logger.info(f"CUDA is available with device: {torch.cuda.get_device_name(0)}")

# Check if federated model exists
federated_model_path = 'federated_adversera_model.pth'
default_model_path = 'optimized_adversera_model.pth'

try:
    if os.path.exists(federated_model_path):
        # Import the load_model function from federated_train
        from federated_train import load_federated_model
        model = load_federated_model(federated_model_path)
        logger.info(f"Loaded federated model from {federated_model_path}")
    elif os.path.exists(default_model_path):
        # Load optimized model if available
        from federated_train import load_federated_model
        model = load_federated_model(default_model_path)
        logger.info(f"Loaded optimized model from {default_model_path}")
    else:
        # Fallback to ImageNet model
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = model.to(device)
        logger.info("Using default ImageNet ResNet50 model")
except Exception as e:
    logger.error(f"Error loading custom model, falling back to default: {str(e)}")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = model.to(device)

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
    'mushroom': ['poisonous', 'non-poisonous'],
    'text': ['informative', 'misleading']  # Simple classification for text images
}


def detect_image_type(image):
    """
    Detect whether the image is a fish eye, mushroom, or text-based image.
    This uses simple heuristics - can be replaced with more sophisticated methods.
    """
    try:
        # Convert image to RGB if it's not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        img_array = np.array(image)

        # Check if this is a text-based image using OCR features
        # For now, we'll use a simple heuristic based on image statistics
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        # Simple analysis based on image statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Calculate edge density (text images typically have high edge density)
        from scipy import ndimage
        edges = ndimage.sobel(gray)
        edge_density = np.mean(edges > 50)  # Threshold for edge detection
        
        # Text images typically have high contrast and many edges
        if edge_density > 0.1 and std_brightness > 60:
            logger.info(f"Detected text image: edge_density={edge_density}, std_brightness={std_brightness}")
            return 'text'
        
        # These thresholds should be adjusted based on your specific use case
        if mean_brightness > 100 and std_brightness < 50:
            return 'fish_eye'
        else:
            return 'mushroom'
    except Exception as e:
        logger.exception(f"Error in detect_image_type: {str(e)}")
        # Default to mushroom as the safest option
        return 'mushroom'


def get_classification(pred_index, image_type):
    """Map prediction index to custom class labels"""
    if image_type in custom_classes:
        return custom_classes[image_type][pred_index % len(custom_classes[image_type])]
    return f"Class {pred_index}"


@app.post("/detect_image_type")
async def detect_image_type_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        detected_type = detect_image_type(image)
        return JSONResponse({
            "image_type": detected_type
        })
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
    logger.info(
        f"Received request: method={method}, epsilon={epsilon}, stealth_mode={stealth_mode}, image_type={image_type}")
    logger.debug(f"All parameters: {locals()}")

    try:
        contents = await file.read()
        logger.info(f"File contents size: {len(contents)} bytes")
        image = Image.open(io.BytesIO(contents))
        logger.info(f"Image opened successfully: {image.format}, {image.size}, {image.mode}")

        # Detect image type if set to auto
        if image_type == 'detect' or image_type == 'auto':
            image_type = detect_image_type(image)

        # Ensure the image is in the correct format (RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Converted image to RGB mode")

        # Preprocess the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)  # Ensure tensor is on the correct device
        logger.info(f"Image preprocessed and moved to device: {device}")

        # Generate adversarial example
        with torch.no_grad():
            try:
                output = model(input_batch)
                logger.info(f"Model forward pass successful, output shape: {output.shape}")
            except Exception as e:
                logger.exception(f"Error during model inference: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")

        original_pred = output.argmax().item()
        logger.info(f"Original prediction index: {original_pred}")

        try:
            logger.info(f"Generating adversarial example with method: {method}, epsilon: {epsilon}")

            # Ensure label tensor is on the same device as the model
            label_tensor = torch.tensor([original_pred], device=device)

            adversarial_image = generate_adversarial_example(
                model, input_batch, label_tensor, method,
                epsilon=epsilon, alpha=alpha, num_iter=num_iter,
                num_classes=num_classes, overshoot=overshoot, max_iter=max_iter,
                pixels=pixels, pop_size=pop_size, delta=delta,
                max_iter_uni=max_iter_uni, max_iter_df=max_iter_df,
                stealth_mode=stealth_mode
            )

            logger.info(f"Adversarial image generated successfully with shape: {adversarial_image.shape}")
        except Exception as e:
            logger.exception(f"Error generating adversarial example: {str(e)}")
            # For text images or other problematic cases, create a simple perturbation
            if "text" in image_type:
                logger.info("Using simple perturbation for text image")
                noise = torch.randn_like(input_batch) * (epsilon / 2)
                adversarial_image = torch.clamp(input_batch + noise, 0, 1)
            else:
                raise HTTPException(status_code=500, detail=f"Failed to generate adversarial example: {str(e)}")

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

        # Get classifications using custom classes
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


@app.get("/models")
async def get_available_models():
    """Get a list of all available models"""
    try:
        logger.info("Fetching available models...")
        
        # Just use the models directory for now to simplify
        models_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(models_dir):
            models_dir = os.getcwd()
        
        logger.info(f"Looking for models in: {models_dir}")
        
        models = []
        # Find .pth files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        
        # Also check the current directory
        if models_dir != os.getcwd():
            model_files += [f for f in os.listdir(os.getcwd()) if f.endswith('.pth') and f not in model_files]
        
        logger.info(f"Found model files: {model_files}")
        
        for file in model_files:
            model_path = os.path.join(models_dir, file) if os.path.exists(os.path.join(models_dir, file)) else os.path.join(os.getcwd(), file)
            
            # Determine model type based on filename
            if "federated" in file.lower():
                model_type = "Federated Learning Model"
            elif "optimized" in file.lower():
                model_type = "Optimized Centralized Model"
            elif "adversera" in file.lower():
                model_type = "AdverseraGuard Model"
            else:
                model_type = "Custom Model"
                
            # Get model stats
            models.append({
                "id": file,
                "name": file.replace('.pth', '').replace('_', ' ').title(),
                "type": model_type,
                "path": model_path,
                "last_modified": os.path.getmtime(model_path),
                "file_size_mb": os.path.getsize(model_path) / (1024 * 1024)
            })
        
        # Add ImageNet model
        models.append({
            "id": "imagenet",
            "name": "ImageNet Default Model",
            "type": "Pre-trained ImageNet Model",
            "path": "N/A",
            "last_modified": 0,
            "file_size_mb": 0
        })
        
        logger.info(f"Returning {len(models)} models: {[m['id'] for m in models]}")
        # Make sure we're returning a proper JSONResponse
        return {"models": models}
    except Exception as e:
        logger.exception(f"Error getting models: {str(e)}")
        # Always return at least the ImageNet model as fallback
        fallback_models = [{
            "id": "imagenet",
            "name": "ImageNet Default Model (Fallback)",
            "type": "Pre-trained ImageNet Model",
            "path": "N/A",
            "last_modified": 0,
            "file_size_mb": 0
        }]
        
        logger.info("Returning fallback ImageNet model due to error")
        return {"models": fallback_models}

@app.get("/model_info")
async def get_model_info():
    """Get information about the currently loaded model and its performance metrics"""
    try:
        # Determine which model is currently loaded
        model_type = "Unknown"
        model_id = os.environ.get("ACTIVE_MODEL", "")
        
        if not model_id:
            # Default model selection logic
            if os.path.exists('federated_adversera_model.pth'):
                model_type = "Federated Learning Model"
                model_path = 'federated_adversera_model.pth'
                model_id = "federated_adversera_model.pth"
            elif os.path.exists('optimized_adversera_model.pth'):
                model_type = "Optimized Centralized Model"
                model_path = 'optimized_adversera_model.pth'
                model_id = "optimized_adversera_model.pth"
            else:
                model_type = "Default ImageNet Model"
                model_path = "N/A"
                model_id = "imagenet"
        else:
            # Model was selected by user
            if model_id == "imagenet":
                model_type = "Default ImageNet Model"
                model_path = "N/A"
            else:
                # Look for model file
                models_dir = os.path.join(os.getcwd(), "models")
                if os.path.exists(os.path.join(models_dir, model_id)):
                    model_path = os.path.join(models_dir, model_id)
                elif os.path.exists(os.path.join(os.getcwd(), model_id)):
                    model_path = os.path.join(os.getcwd(), model_id)
                else:
                    model_path = "N/A"
                    
                # Determine model type
                if "federated" in model_id.lower():
                    model_type = "Federated Learning Model"
                elif "optimized" in model_id.lower():
                    model_type = "Optimized Centralized Model"
                else:
                    model_type = "Custom Model"
            
        # Get model stats
        model_stats = {
            "model_id": model_id,
            "model_type": model_type,
            "model_path": model_path,
            "device": str(next(model.parameters()).device),
            "using_cuda": torch.cuda.is_available(),
        }
        
        # Add additional metrics if it's a custom model
        if model_type != "Default ImageNet Model" and os.path.exists(model_path):
            # Get model creation/modification time
            model_stats["last_modified"] = os.path.getmtime(model_path)
            model_stats["file_size_mb"] = os.path.getsize(model_path) / (1024 * 1024)

        return JSONResponse(model_stats)
    except Exception as e:
        logger.exception(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/select_model")
async def select_model(model_id: str = Form(...)):

    """Select a model to use"""
    try:
        # Check if model exists
        if model_id != "imagenet":
            # Look for model file
            models_dir = os.path.join(os.getcwd(), "models") 
            model_path = ""
            
            if os.path.exists(os.path.join(models_dir, model_id)):
                model_path = os.path.join(models_dir, model_id)
            elif os.path.exists(os.path.join(os.getcwd(), model_id)):
                model_path = os.path.join(os.getcwd(), model_id)
                
            if not model_path or not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
                
            # Load model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            try:
                # Try loading as EfficientNet
                from federated_train import load_federated_model
                logger.info(f"Loading model {model_id} as EfficientNet")
                loaded_model = load_federated_model(model_path)
            except Exception as e:
                logger.error(f"Failed to load as EfficientNet: {str(e)}")
                # Try loading as ResNet
                logger.info(f"Loading model {model_id} as ResNet")
                loaded_model = resnet50(weights=None)
                loaded_model.load_state_dict(torch.load(model_path, map_location=device))
            

        
        # Set active model in environment
        os.environ["ACTIVE_MODEL"] = model_id
        
        logger.info(f"Model {model_id} selected and loaded successfully")
        return JSONResponse({"status": "success", "message": f"Model {model_id} selected"})
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Error selecting model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/federated_training_status")
async def get_federated_training_status():
    """Check status of federated learning training"""
    try:
        federated_model_path = 'federated_adversera_model.pth'
        if not os.path.exists(federated_model_path):
            models_dir = os.path.join(os.getcwd(), "models")
            federated_model_path = os.path.join(models_dir, 'federated_adversera_model.pth')
            if not os.path.exists(federated_model_path):
                return JSONResponse({
                    "trained": False,
                    "message": "No federated model found. Run federated training first."
                })
            
        # If we have a model, return info about it
        model_creation_time = os.path.getctime(federated_model_path)
        model_modified_time = os.path.getmtime(federated_model_path)
        
        return JSONResponse({
            "trained": True,
            "model_path": federated_model_path,
            "model_id": os.path.basename(federated_model_path),
            "created_at": model_creation_time,
            "last_modified": model_modified_time,
            "file_size_mb": os.path.getsize(federated_model_path) / (1024 * 1024),
            "message": "Federated model is trained and ready to use."
        })
    except Exception as e:
        logger.exception(f"Error checking federated training status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate_model")
async def evaluate_model(
    file: UploadFile = File(...),
    model_id: str = Form(...),
    attack_method: str = Form(...),
    epsilon: float = Form(0.03)
):
    """Evaluate a specific model against an adversarial attack"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Detect image type
        image_type = detect_image_type(image)
        
        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Load the model
        loaded_model = load_model(model_id)
        
        # Get initial prediction
        with torch.no_grad():
            output = loaded_model(input_batch)
        initial_pred = output.argmax().item()
        
        # Evaluate against single attack method
        results = evaluate_model_performance(
            loaded_model, 
            input_batch,
            torch.tensor([initial_pred]),
            [attack_method],
            epsilon
        )
        
        # Get classifications using custom classes
        for attack, data in results.items():
            if "prediction" in data:
                data["class"] = get_classification(data["prediction"], image_type)
        
        # Create response with base64 images
        response = {
            "model_id": model_id,
            "model_name": model_id.replace('.pth', '').replace('_', ' ').title() if model_id != "imagenet" else "ImageNet Default Model",
            "attack_method": attack_method,
            "results": results,
            "original_class": get_classification(initial_pred, image_type)
        }
        
        # Generate adversarial image for visualization
        if attack_method in results and "error" not in results[attack_method]:
            # Create adversarial example for display
            adv_image = generate_adversarial_example(
                loaded_model, 
                input_batch,
                torch.tensor([initial_pred]), 
                attack_method,
                epsilon=epsilon
            )
            
            # Convert tensor to PIL Image
            to_pil = transforms.ToPILImage()
            adv_image_pil = to_pil(adv_image.squeeze(0))
            
            # Save to bytes and encode
            img_byte_arr = io.BytesIO()
            adv_image_pil.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
            
            response["adversarial_image"] = img_base64
            
        return JSONResponse(response)
    
    except Exception as e:
        logger.exception(f"Error evaluating model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Shared functions for model evaluation
def load_model(model_id: str) -> torch.nn.Module:
    """Load a model by its ID"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_id == "imagenet":
            return resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
        
        # Look for model file
        models_dir = os.path.join(os.getcwd(), "models") 
        model_path = ""
        
        if os.path.exists(os.path.join(models_dir, model_id)):
            model_path = os.path.join(models_dir, model_id)
        elif os.path.exists(os.path.join(os.getcwd(), model_id)):
            model_path = os.path.join(os.getcwd(), model_id)
            
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model {model_id} not found")
            
        try:
            # Try loading as EfficientNet
            from federated_train import load_federated_model
            loaded_model = load_federated_model(model_path)
        except Exception as e:
            # Try loading as ResNet
            loaded_model = resnet50(weights=None)
            loaded_model.load_state_dict(torch.load(model_path, map_location=device))
        
        loaded_model = loaded_model.to(device)
        loaded_model.eval()
        return loaded_model
    
    except Exception as e:
        logger.exception(f"Error loading model {model_id}: {str(e)}")
        raise ValueError(f"Failed to load model {model_id}: {str(e)}")


def evaluate_model_performance(model: torch.nn.Module, 
                              image: torch.Tensor,
                              label: torch.Tensor,
                              attack_methods: List[str] = ["fgsm", "pgd", "deepfool"],
                              epsilon: float = 0.03) -> Dict[str, Any]:
    """Evaluate model performance against various attack methods"""
    results = {}
    device = next(model.parameters()).device
    image = image.to(device)
    label = label.to(device)
    
    # Original prediction
    with torch.no_grad():
        output = model(image)
    
    original_pred = output.argmax().item()
    original_confidence = torch.nn.functional.softmax(output, dim=1)[0, original_pred].item()
    
    results["original"] = {
        "prediction": original_pred,
        "confidence": original_confidence,
        "success_rate": 1.0 if original_pred == label.item() else 0.0
    }
    
    # Test against each attack method
    for method in attack_methods:
        try:
            start_time = time.time()
            
            # Generate adversarial example
            if method == "fgsm":
                perturbed_image = fgsm_attack(model, image, label, epsilon)
            elif method == "pgd":
                perturbed_image = pgd_attack(model, image, label, epsilon)
            elif method == "deepfool":
                perturbed_image = deepfool_attack(model, image, 10)  # Using 10 classes for speed
            elif method == "one_pixel":
                perturbed_image = one_pixel_attack(model, image.squeeze(0), label, pixels=1)
                perturbed_image = perturbed_image.unsqueeze(0)
            else:
                continue
                
            generation_time = time.time() - start_time
            
            # Get prediction for adversarial image
            with torch.no_grad():
                adv_output = model(perturbed_image)
            
            adv_pred = adv_output.argmax().item()
            adv_confidence = torch.nn.functional.softmax(adv_output, dim=1)[0, adv_pred].item()
            
            results[method] = {
                "prediction": adv_pred,
                "confidence": adv_confidence,
                "attack_success": original_pred != adv_pred,
                "time_taken": generation_time
            }
            
        except Exception as e:
            logger.exception(f"Error evaluating model with {method} attack: {str(e)}")
            results[method] = {
                "error": str(e)
            }
    
    return results


@app.post("/compare_models")
async def compare_models(
    file: UploadFile = File(...),
    model_ids: List[str] = Body(...),
    attack_methods: List[str] = Body(["fgsm", "pgd", "deepfool"]),
    epsilon: float = Body(0.03)
):
    """Compare multiple models against various adversarial attacks"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Load and evaluate each model
        all_results = {}
        
        for model_id in model_ids:
            try:
                # Load the model
                logger.info(f"Loading model {model_id} for comparison")
                loaded_model = load_model(model_id)
                
                # Detect image type
                image_type = detect_image_type(image)
                
                # Get initial prediction to use as label
                with torch.no_grad():
                    output = loaded_model(input_batch)
                initial_pred = output.argmax().item()
                
                # Evaluate performance
                results = evaluate_model_performance(
                    loaded_model, 
                    input_batch,
                    torch.tensor([initial_pred]),
                    attack_methods,
                    epsilon
                )
                
                # Get classifications using custom classes
                for attack, data in results.items():
                    if "prediction" in data:
                        data["class"] = get_classification(data["prediction"], image_type)
                
                # Add to overall results
                all_results[model_id] = results
                
            except Exception as e:
                logger.exception(f"Error evaluating model {model_id}: {str(e)}")
                all_results[model_id] = {"error": str(e)}
        
        # Process results into comparison metrics
        comparison = {
            "models": {},
            "attack_methods": attack_methods,
            "overall_ranking": []
        }
        
        # Calculate defense scores (higher is better)
        for model_id, results in all_results.items():
            model_name = model_id.replace('.pth', '').replace('_', ' ').title()
            if model_id == "imagenet":
                model_name = "ImageNet Default Model"
                
            success_count = 0
            total_attacks = 0
            avg_confidence = 0
            avg_time = 0
            
            for method in attack_methods:
                if method in results and "attack_success" in results[method]:
                    total_attacks += 1
                    # For defense score, we want to count when the attack FAILED
                    if not results[method]["attack_success"]:
                        success_count += 1
                    
                    avg_confidence += results[method].get("confidence", 0)
                    avg_time += results[method].get("time_taken", 0)
            
            defense_score = (success_count / total_attacks) * 100 if total_attacks > 0 else 0
            avg_confidence = avg_confidence / total_attacks if total_attacks > 0 else 0
            avg_time = avg_time / total_attacks if total_attacks > 0 else 0
            
            comparison["models"][model_id] = {
                "name": model_name,
                "defense_score": defense_score,
                "attack_resistance": success_count,
                "attacks_evaluated": total_attacks,
                "avg_confidence": avg_confidence,
                "avg_response_time": avg_time,
                "detailed_results": results
            }
            
            comparison["overall_ranking"].append({
                "model_id": model_id,
                "name": model_name,
                "defense_score": defense_score
            })
        
        # Sort the ranking by defense score
        comparison["overall_ranking"] = sorted(
            comparison["overall_ranking"], 
            key=lambda x: x["defense_score"], 
            reverse=True
        )
            
        return JSONResponse(comparison)
    
    except Exception as e:
        logger.exception(f"Error comparing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Debug routes to help diagnose issues
@app.get("/")
async def root():
    """Root endpoint for testing"""
    return {"message": "AdverseraGuard API is running", "version": "2.0"}

@app.get("/routes")
async def list_routes():
    """List all available routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name
            })
    return {"routes": routes}

@app.get("/debug/models")
async def debug_models():
    """Debug endpoint for models directory"""
    models_dir = os.path.join(os.getcwd(), "models")
    current_dir = os.getcwd()
    
    model_files_in_models = []
    model_files_in_current = []
    
    if os.path.exists(models_dir):
        model_files_in_models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        
    model_files_in_current = [f for f in os.listdir(current_dir) if f.endswith('.pth')]
    
    return {
        "cwd": current_dir,
        "models_dir": models_dir,
        "models_dir_exists": os.path.exists(models_dir),
        "model_files_in_models_dir": model_files_in_models,
        "model_files_in_current_dir": model_files_in_current
    }

if __name__ == "__main__":
    import uvicorn
    
    # Print out the application info
    print(f"Starting AdverseraGuard API server")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Models directory: {os.path.join(os.getcwd(), 'models')}")
    print(f"Available models: {[f for f in os.listdir(os.path.join(os.getcwd(), 'models')) if f.endswith('.pth')] if os.path.exists(os.path.join(os.getcwd(), 'models')) else []}")
    
    # Use reload=False to bypass the file scanning issue
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=False, 
        log_level="info"
    )