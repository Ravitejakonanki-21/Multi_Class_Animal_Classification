Multi_Class_Animal_Classification

Project Description
This project implements a multi-class deep learning model for animal image classification. The system takes an input image of an animal and predicts which category it belongs to. The project includes dataset handling, preprocessing, model training, evaluation, and inference.

Model Download
Trained model file (Google Drive):
https://drive.google.com/file/d/1iD2qOb18Caa1dIRua161dRxHbxSC6NWW/view?usp=sharing

Features
- Multi-class animal image classification
- CNN-based architecture
- Training and validation pipeline
- Inference script to classify new images
- Model file available for direct download
- Easy-to-understand folder structure

Repository Structure
Multi_Class_Animal_Classification
│
├── data (dataset folder)
├── src
│   ├── train.py
│   ├── model.py
│   ├── predict.py
│   ├── utils.py
├── notebooks
├── saved_models
├── requirements.txt
└── README.txt

Tech Stack
- Python 3
- TensorFlow or Keras
- NumPy
- OpenCV or PIL
- Matplotlib
- Scikit-learn

Installation
Clone the repository:
git clone https://github.com/Ravitejakonanki-21/Multi_Class_Animal_Classification.git
cd Multi_Class_Animal_Classification

Install dependencies:
pip install -r requirements.txt

Dataset
Dataset should be organized as follows:

data
├── class_name_1
├── class_name_2
├── class_name_3
└── more_classes

Training
python src/train.py

Evaluation
python src/evaluate.py

Prediction
python src/predict.py --image path/to/image.jpg

Future Enhancements
- Add web interface (Flask, FastAPI, Streamlit)
- Add transfer learning models
- Add more classes and augmentation
- Deploy using Docker or cloud

Author
Raviteja Konanki
GitHub: https://github.com/Ravitejakonanki-21

