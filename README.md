# AdverseraGuard V2

A platform for testing and evaluating adversarial attacks on deep learning models, with a focus on image classification.

## Features

- Test various adversarial attack methods (FGSM, PGD, DeepFool, One Pixel Attack, UAP)
- Visualize original and adversarial images
- Support for different model training approaches (centralized and federated learning)
- Stealth mode for more subtle adversarial perturbations
- Support for different image types (mushroom classification, fish freshness)

## Project Structure

```
├── backend/                      # Backend server and ML functionality
│   ├── adversarial_methods.py    # Implementation of attack algorithms
│   ├── main.py                   # FastAPI server
│   ├── train.py                  # Centralized model training
│   ├── federated_server.py       # Federated learning server
│   ├── federated_client.py       # Federated learning client
│   ├── federated_train.py        # Dataset splitting for federated learning
│   ├── federated_main.py         # Main script to run federated learning
│   ├── dataset/                  # Training and validation data
│   └── requirements.txt          # Python dependencies
├── frontend/                     # React web interface
│   ├── public/                   # Static files
│   ├── src/                      # React components
│   └── package.json              # Frontend dependencies
└── README.md                     # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- PyTorch and TorchVision
- Flower (for federated learning)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/AdverseraGuardV2.git
cd AdverseraGuardV2
```

2. Install backend dependencies:

```bash
cd backend
pip install -r updated_requirements.txt
```

3. Install frontend dependencies:

```bash
cd ../frontend
npm install
```

### Running the Application

1. Start the backend server:

```bash
cd backend
python main.py
```

2. Start the frontend development server:

```bash
cd frontend
npm start
```

3. Access the web interface at http://localhost:3000

## Federated Learning Support

AdverseraGuardV2 now supports federated learning for privacy-preserving model training.

### Running Federated Learning

1. Start federated learning process:

```bash
cd backend
python federated_main.py --num_clients 3 --num_rounds 5 --compare
```

This will:
- Split your dataset into client partitions
- Start the federated learning server
- Launch multiple client processes
- Train a federated model
- Compare the federated model with the centralized one

2. Test an individual image against the federated model:

```bash
python federated_adversarial_test.py \
    --federated_model federated_adversera_model.pth \
    --image dataset/val/poisonous/example.jpg \
    --attack fgsm
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
