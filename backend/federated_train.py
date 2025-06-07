import os
# Set OpenMP environment variable to fix the duplication error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torchvision import models
import argparse
import logging
import shutil
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("federated_train")

def setup_federated_dataset(data_dir, num_clients=2, validation_split=0.2):
    """
    Splits the dataset into multiple client partitions for federated learning.
    Creates a directory structure suitable for federated training.
    
    Args:
        data_dir: Path to the original dataset with train/val subdirectories
        num_clients: Number of federated clients to create
        validation_split: Portion of each client's data to use for validation
    """
    # Ensure we have the correct dataset path
    data_path = Path(data_dir)
    
    # Check if the dataset directory exists
    if not data_path.exists():
        # Try looking for dataset in the current directory
        possible_paths = [
            Path("dataset"),
            Path("backend/dataset"),
            Path("../dataset"),
            Path(os.getcwd()) / "dataset"
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "train").exists():
                data_path = path
                logger.info(f"Found dataset at {data_path}")
                break
        else:
            logger.error(f"Could not find dataset directory in {data_dir} or any standard locations")
            raise FileNotFoundError(f"Dataset directory not found in {data_dir}")
    
    logger.info(f"Using dataset at: {data_path}")
    
    # Create main federated directory
    federated_dir = data_path.parent / "federated_dataset"
    os.makedirs(federated_dir, exist_ok=True)
    
    # Get list of classes from original dataset
    train_dir = data_path / "train"
    if not train_dir.exists():
        logger.error(f"Train directory not found at {train_dir}")
        raise FileNotFoundError(f"Train directory not found at {train_dir}")
        
    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    logger.info(f"Found classes: {classes}")
    
    # For each client, create a directory structure and allocate data
    for client_id in range(1, num_clients + 1):
        client_dir = federated_dir / f"client_{client_id}"
        
        # Create train and val directories for this client
        client_train_dir = client_dir / "train"
        client_val_dir = client_dir / "val"
        
        os.makedirs(client_train_dir, exist_ok=True)
        os.makedirs(client_val_dir, exist_ok=True)
        
        # Create class subdirectories
        for class_name in classes:
            os.makedirs(client_train_dir / class_name, exist_ok=True)
            os.makedirs(client_val_dir / class_name, exist_ok=True)
    
    # Distribute data among clients
    for class_name in classes:
        # Get all files for this class
        class_files = list((train_dir / class_name).glob("*.jpg"))
        random.shuffle(class_files)
        
        # Calculate number of files per client (approximately equal)
        files_per_client = len(class_files) // num_clients
        
        for client_id in range(1, num_clients + 1):
            # Determine files for this client
            start_idx = (client_id - 1) * files_per_client
            end_idx = start_idx + files_per_client if client_id < num_clients else len(class_files)
            client_files = class_files[start_idx:end_idx]
            
            # Split into train and validation
            val_size = int(len(client_files) * validation_split)
            train_files = client_files[val_size:]
            val_files = client_files[:val_size]
            
            # Copy files to client directories
            client_train_class_dir = federated_dir / f"client_{client_id}" / "train" / class_name
            client_val_class_dir = federated_dir / f"client_{client_id}" / "val" / class_name
            
            for src_file in train_files:
                dst_file = client_train_class_dir / src_file.name
                shutil.copy2(src_file, dst_file)
            
            for src_file in val_files:
                dst_file = client_val_class_dir / src_file.name
                shutil.copy2(src_file, dst_file)
    
    # Copy validation data from original dataset to each client's validation directory
    # This ensures clients have the same evaluation data
    val_dir = Path(data_dir) / "val"
    if val_dir.exists():
        for class_name in classes:
            val_class_dir = val_dir / class_name
            if val_class_dir.exists():
                val_files = list(val_class_dir.glob("*.jpg"))
                
                # Choose a subset of validation files to copy to each client
                val_subset = random.sample(val_files, min(len(val_files), 5))
                
                for client_id in range(1, num_clients + 1):
                    client_val_class_dir = federated_dir / f"client_{client_id}" / "val" / class_name
                    
                    for src_file in val_subset:
                        dst_file = client_val_class_dir / src_file.name
                        if not dst_file.exists():  # Avoid overwriting files from the train split
                            shutil.copy2(src_file, dst_file)
    
    # Log dataset statistics
    logger.info("Federated dataset created")
    for client_id in range(1, num_clients + 1):
        client_dir = federated_dir / f"client_{client_id}"
        train_count = sum(len(list((client_dir / "train" / c).glob("*.jpg"))) for c in classes)
        val_count = sum(len(list((client_dir / "val" / c).glob("*.jpg"))) for c in classes)
        
        logger.info(f"Client {client_id}: {train_count} training images, {val_count} validation images")
    
    return str(federated_dir)

def load_federated_model(model_path):
    """
    Load the federated model trained using Flower.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create model architecture
    model = models.efficientnet_b3(pretrained=False)
    num_classes = 4  # Same as in train.py
    
    # Modify the classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    
    # Load model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded federated model from {model_path}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated dataset setup")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Original dataset directory")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of federated clients")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation data portion")
    args = parser.parse_args()
    
    federated_dir = setup_federated_dataset(args.data_dir, args.num_clients, args.val_split)