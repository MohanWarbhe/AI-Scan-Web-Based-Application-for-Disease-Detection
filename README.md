# 🧠 AI-Scan – Web-Based Application for Disease Detection

<p align="center">
  <img src="https://img.shields.io/badge/AI-Healthcare-blue?style=for-the-badge&logo=google-health">
  <img src="https://img.shields.io/badge/Machine%20Learning-TensorFlow-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Python-Backend-yellow?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Web%20App-Flask-green?style=for-the-badge&logo=flask">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge">
</p>

<p align="center">
  🚑 An AI-powered web application for early disease detection using medical images.
</p>

<p align="center">
  <b>Major Project | AI • ML • Healthcare • Web Application</b>
</p>

---

## 📌 Project Overview

**AI-Scan** is a web-based healthcare application that uses  
**Artificial Intelligence & Deep Learning** to detect diseases from medical images.

The system allows users to upload medical scans and receive **fast, accurate predictions**, helping in **early diagnosis and decision support**.

This project demonstrates the real-world application of **AI in medical healthcare**.

---

## ✨ Key Features

- 🧠 AI-based disease detection  
- 📷 Medical image upload & analysis  
- 📊 Prediction with confidence score  
- 🌐 Clean & user-friendly web interface  
- ⚡ Fast processing using trained ML model  
- 🔐 Secure backend handling  

---

## 🛠️ Tech Stack

| Layer | Technology |
|------|-----------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| AI / ML | TensorFlow, Keras |
| Image Processing | OpenCV |
| Tools | Git, GitHub |
| Environment | Virtual Environment |

---

## 📂 Project Structure

AI-Scan-Web-Based-Application-for-Disease-Detection/
│
├── app.py # Main Flask application
├── model/ # Trained ML model
├── templates/ # HTML files
├── static/ # CSS, JS, images
├── utils/ # Helper functions
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files
└── README.md # Project documentation


---

## 🔄 Application Workflow (How It Works)

1️⃣ **User uploads medical image**  
2️⃣ Image is **preprocessed** (resize, normalization)  
3️⃣ Image is passed to **trained ML model**  
4️⃣ Model predicts disease class  
5️⃣ Result is displayed on the web interface  

---

## 🧠 Core Prediction Logic (Sample Code)

python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("model/disease_model.h5")

def predict_disease(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))
    prediction = model.predict(img)
    return prediction

## 👨‍💻 Author & Contact

**Mohan Warbhe**  
🎓 Computer Science & Design  
💡 AI | Machine Learning | Python | Web Development  

📧 **Email:** [mohan.warbhe.work@gmail.com](mailto:mohan.warbhe.work@gmail.com)  

🔗 **GitHub:** https://github.com/MohanWarbhe



# Clone repository
git clone https://github.com/MohanWarbhe/AI-Scan-Web-Based-Application-for-Disease-Detection.git

# Go to project directory
cd AI-Scan-Web-Based-Application-for-Disease-Detection

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py





