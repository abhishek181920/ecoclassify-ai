EcoClassify AI – Waste Classification System
Project Overview
EcoClassify AI is an AI-powered waste classification system leveraging Convolutional Neural Networks (CNNs) to classify waste images into six categories: plastic, metal, glass, cardboard, paper, and trash. The system enables real-time waste recognition via webcam or image upload and is deployed as an interactive web/mobile app using Streamlit.

Problem Statement
Objective:
Develop an AI-powered waste classification system using CNNs to classify images of waste into six core categories: plastic, metal, glass, cardboard, paper, and trash. The project aims to simplify waste sorting and promote recycling with a user-friendly mobile/web application for real-time image-based classification.

Features
Instant Waste Classification:
Upload or capture waste images using your device’s camera and get fast AI-based categorization.

Supported Classes:

📦 Cardboard

🥛 Glass

🔧 Metal

📄 Paper

🥤 Plastic

🗑️ Trash

Real-time Prediction:
Classifies images in real-time using a pre-trained deep learning model (ResNet50V2 by default, EfficientNetV2B1 as alternative).

Detailed Results:

Main predicted class and confidence score

Bar chart of confidence scores for all categories

Personalized recycling tips for the detected class

Environmental impact suggestions

Intuitive UI:

Clean Streamlit interface with custom CSS

Interactive sidebar with app information, user instructions, model stats, and confidence threshold slider

Technical Highlights
Deep Learning Model:
ResNet50V2 (or EfficientNetV2B1) trained via TensorFlow/Keras.
Model optimizations include data augmentation, label smoothing, and Mixed Precision (FP16) GPU support for high speed and efficiency.

Data Preprocessing:

Automated resizing, color normalization, and augmentation for each upload

MixUp regularization for robust generalization

Real-Time Deployment:
Streamlit app (app.py) lets users upload or capture photos, view classification results, and access recycling guidance—all directly in the browser.

Model Evaluation Metrics:

Accuracy:  0.8436 (84.36%)
Precision: 0.8465 (84.65%)
Recall:    0.8436 (84.36%)
F1-Score:  0.8402 (84.02%)

classification report:
               precision   recall   f1-score   support

   cardboard     0.8421    0.9275    0.8828        69
       glass     0.8447    0.8529    0.8488       102
       metal     0.7593    0.9318    0.8367        88
       paper     0.9187    0.9187    0.9187       123
     plastic     0.8714    0.6854    0.7673        89
       trash     0.7600    0.5588    0.6441        34

    accuracy                         0.8436       505
   macro avg     0.8327    0.8125    0.8164       505
weighted avg     0.8465    0.8436    0.8402       505

How to Use
Launch the App:
Run the following command in your terminal

text
streamlit run app.py
Upload or Take a Photo:
Use the provided UI to upload an image or capture one using your webcam.

Classify Waste:
Click "Classify Waste" to receive instant prediction, confidence levels, and actionable recycling tips.

Review Results:
Explore confidence for all classes in the bar chart, understand environmental impact, and review model details in sidebar.

Requirements

Python 3.10+ (Tested on Windows OS)
TensorFlow 2.x
Streamlit
NumPy
Pandas
Pillow
Matplotlib
OpenCV-Python

(Optional) GPU with Mixed Precision enabled
Install all dependencies via:

text
pip install tensorflow streamlit numpy pandas pillow matplotlib opencv-python
File Structure
app.py – Streamlit web app for real-time classification
resnet50v2_waste_classifier_best.keras or eco_classify_ai_model.keras – Pre-trained model files
README.md – Project documentation
requirements.txt – Python dependencies
Model Training (Optional)
Train your own ResNet50V2 waste classifier using the provided training scripts. Customize hyperparameters and dataset path as required.

Acknowledgments
TrashNet Dataset from Kaggle
TensorFlow & Keras community
Streamlit framework for rapid prototyping
