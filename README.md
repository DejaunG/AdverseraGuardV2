# 🛡️ AdversaGuard: Adversarial Attack Testing Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.0%2B-blue)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)](https://fastapi.tiangolo.com/)

AdversaGuard is a powerful web-based platform for testing and visualizing adversarial attacks on deep learning models. It provides an intuitive interface for generating and analyzing adversarial examples using various attack methods.

## 🌟 Features

- 🎯 Multiple adversarial attack methods:
  - Fast Gradient Sign Method (FGSM)
  - Projected Gradient Descent (PGD)
  - DeepFool
  - One Pixel Attack
  - Universal Adversarial Perturbations

- 🔧 Advanced configuration options for each attack method
- 📊 Real-time visualization of original and adversarial images
- 🤖 Automatic image type detection
- 🎨 Modern, responsive UI built with Material-UI
- 🚀 Fast and efficient backend powered by FastAPI

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 14.0 or higher
- npm 6.0 or higher

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/adversaguard.git
cd adversaguard

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
cd backend
pip install -r requirements.txt
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Create production build
npm run build
```

## 🚀 Usage

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

3. Open your browser and navigate to `http://localhost:3000`

## 💡 How It Works

1. Upload an image (supported types: fish eye images or mushroom images)
2. Select an attack method and configure parameters
3. Generate adversarial examples
4. View the results and classifications

## 🔧 Configuration

### Attack Parameters

- `epsilon`: Perturbation magnitude (default: 0.03)
- `alpha`: Step size for iterative attacks (default: 0.01)
- `num_iter`: Number of iterations (default: 40)
- And more advanced parameters...

### Supported Image Types

- 🐟 Fish Eye Images (Classification: Fresh/Non-Fresh)
- 🍄 Mushroom Images (Classification: Poisonous/Non-Poisonous)

## 📚 API Documentation

The backend API is documented using FastAPI's automatic documentation. After starting the backend server, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🌟 Acknowledgments

- FastAPI for the amazing backend framework
- React and Material-UI for the frontend components
- PyTorch for the deep learning capabilities

## 📞 Contact

Website - [FutureForge.xyz](https://www.futureforge.xyz/)

Project Link: [https://github.com/DejaunG/AdverseraGuardV2](https://github.com/DejaunG/AdverseraGuardV2)

---

<p align="center">
  Made with ❤️ by [Dejaun Gayle]
</p>