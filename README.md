AdverseraGuard
AdverseraGuard is a web application that demonstrates adversarial attacks on image classification models. Users can upload images of fish eyes, select an adversarial attack method (FGSM or PGD), generate adversarial images, and see the classification results using a fine-tuned MobileNetV2 model.

Table of Contents
Features
Project Structure
Prerequisites
Setup Instructions
Backend Setup
Frontend Setup
Running the Application
Usage
Troubleshooting
Contributing
License
Features
Upload images of fish eyes (fresh or non-fresh).
Select between FGSM and PGD adversarial attack methods.
Generate adversarial images.
Classify images using a fine-tuned MobileNetV2 model.
Display original and adversarial images along with classification results.
Project Structure
kotlin
Copy code
adverseraguardv2/
├── backend/
│   ├── dataset/
│   │   ├── train/
│   │   │   ├── Edible/
│   │   │   ├── Fresh/
│   │   │   └── Non-fresh/
│   │   │   ├── Poisonous/
│   │   └── val/
│   │       ├── Edible/
│   │   │   ├── Fresh/
│   │   │   └── Non-fresh/
│   │   │   ├── Poisonous/
│   ├── venv/
│   ├── main.py
│   ├── adversarial_methods.py
│   ├── train.py
│   ├── fish_eye_model.pth
│   └── requirements.txt
└── frontend/
    ├── node_modules/
    ├── public/
    ├── src/
    │   ├── components/
    │   ├── utils/
    │   ├── App.js
    │   ├── index.js
    │   └── index.css
    ├── package.json
    └── package-lock.json
Prerequisites
Python 3.8 or higher
Node.js and npm
Virtual Environment tools (venv)
Setup Instructions
Backend Setup
Navigate to the backend directory:

bash
Copy code
cd adverseraguardv2/backend
Create and activate a virtual environment:

For Windows (PowerShell):

powershell
Copy code
python -m venv venv
.\venv\Scripts\Activate.ps1
For Windows (Command Prompt):

cmd
Copy code
python -m venv venv
venv\Scripts\activate.bat
For macOS/Linux:

bash
Copy code
python3 -m venv venv
source venv/bin/activate
Install Python dependencies:

bash
Copy code
pip install -r requirements.txt
Prepare the dataset:

Place your images into the corresponding folders in backend/dataset/train/ and backend/dataset/val/.
Ensure that images are organized under Fresh and Non-fresh subdirectories.
Train the model:

bash
Copy code
python train.py
This will generate fish_eye_model.pth in the backend directory.
Run the backend server:

bash
Copy code
uvicorn main:app --reload
The server will be accessible at http://localhost:8000.
Frontend Setup
Navigate to the frontend directory:

bash
Copy code
cd adverseraguardv2/frontend
Install Node.js dependencies:

bash
Copy code
npm install
Start the React application:

bash
Copy code
npm start
The application will open in your browser at http://localhost:3000.
Running the Application
Ensure both the backend and frontend servers are running.
Access the application by visiting http://localhost:3000 in your web browser.
Usage
Upload an Image:

Click on "Upload Image" and select a fish eye image from your computer.
Select Attack Method:

Choose either "FGSM" or "PGD" from the dropdown menu.
Generate Adversarial Image:

Click on "Generate Adversarial Image".
View Results:

The original and adversarial images will be displayed.
The classification result will be shown below the images.
Troubleshooting
An error occurred while processing the image:

Check the backend console for error messages.
Ensure all dependencies are installed.
Verify that the model is correctly loaded and the paths are correct.
CORS Issues:

If you encounter CORS errors, ensure the CORS middleware is configured in main.py.
Backend Not Responding:

Ensure the backend server is running on http://localhost:8000.
Check for any errors in the backend console.
Frontend Not Loading:

Ensure the frontend development server is running.
Check the browser console for errors.
Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.

License
This project is licensed under the MIT License.

Additional Notes
Security Considerations:

In a production environment, update the CORS configuration to allow only trusted origins.
Validate and sanitize all inputs to prevent security vulnerabilities.
Data Privacy:

Ensure that any uploaded images are handled securely and are not stored longer than necessary.
Performance Optimization:

For better performance, consider using a production ASGI server like Gunicorn with Uvicorn workers.
Use a reverse proxy like Nginx to manage requests.
