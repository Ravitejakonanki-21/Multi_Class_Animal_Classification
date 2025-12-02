Multi_Class_Animal_Classification
Project Description
This project implements a multi-class deep learning model for animal image classification. The system takes an input image of an animal and predicts which category it belongs to. The project includes dataset handling, preprocessing, model training, evaluation, and inference.
Model Download
Trained model file (Google Drive):
https://drive.google.com/file/d/1iD2qOb18Caa1dIRua161dRxHbxSC6NWW/view?usp=sharing
Features
Multi-class animal image classification
CNN-based architecture
Training and validation pipeline
Inference script to classify new images
Model file available for direct download
Easy-to-understand folder structure
Repository Structure
Multi_Class_Animal_Classification
│
├── data (dataset folder)
├── src
│ ├── train.py
│ ├── model.py
│ ├── predict.py
│ ├── utils.py
├── notebooks
├── saved_models
├── requirements.txt
└── README.md
Tech Stack
Python 3
TensorFlow or Keras
NumPy
OpenCV or PIL
Matplotlib
Scikit-learn
Installation
Clone the repository:
git clone https://github.com/Ravitejakonanki-21/Multi_Class_Animal_Classification.git
cd Multi_Class_Animal_Classification
Install dependencies:
pip install -r requirements.txt
Dataset
The dataset should be organized as follows:
data
├── class_name_1
├── class_name_2
├── class_name_3
└── more_classes
Each folder contains sample images of that class.
Training
Run the training script:
python src/train.py
This trains the CNN model and stores it in the saved_models directory.
Evaluation
Run evaluation:
python src/evaluate.py
Prediction
Run prediction on your image:
python src/predict.py --image path/to/image.jpg
Or in Python code:
from src.predict import predict_animal_class
print(predict_animal_class("image.jpg"))
Future Enhancements
Add a web interface using Flask, FastAPI, or Streamlit
Use transfer learning models like ResNet or EfficientNet
Add more classes and data augmentation
Deploy using Docker or cloud platforms
Author
Raviteja Konanki
GitHub: https://github.com/Ravitejakonanki-21
