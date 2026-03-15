# 🩺 Mediscan: A Deep Learning Based Disease Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![React](https://img.shields.io/badge/Frontend-React-blue)
![Deep Learning](https://img.shields.io/badge/AI-Deep%20Learning-orange)


Mediscan is an **AI-powered medical image analysis platform** designed to assist in the **early detection of diseases using deep learning models**.

The system allows users to upload medical images and receive **AI-generated predictions** for multiple diseases through an interactive web interface.

The platform integrates:

* **Deep Learning models for disease detection**
* **FastAPI backend for high-performance APIs**
* **React frontend for user interaction**

Currently, Mediscan supports detection for:

* 🧴 Skin Diseases
* 🧠 Brain Tumors
* 🦠 Malaria Parasites

---

# 🚀 Features

✔ AI-powered medical image classification
✔ Multi-disease detection system
✔ Fast REST API using FastAPI
✔ Modern frontend built with React
✔ Modular architecture for adding new diseases
✔ Real-time predictions
✔ Deep learning–based medical analysis

---

# 🏗 System Architecture

```
            User
             │
             ▼
      React Frontend
             │
             ▼
        FastAPI Server
             │
   ┌─────────┼─────────┐
   ▼         ▼         ▼
Skin Model  Brain Model  Malaria Model
   │         │         │
   ▼         ▼         ▼
      Prediction Result
             │
             ▼
       Display in UI
```

---

# 🧠 AI Models

## 1️⃣ Malaria Detection

**Goal:** Detect malaria parasites in blood smear images.

### Model

MobileNetV2 (Transfer Learning)

### Configuration

| Parameter  | Value                   |
| ---------- | ----------------------- |
| Image Size | 128 × 128               |
| Optimizer  | Adam                    |
| Loss       | Binary Crossentropy     |
| Epochs     | 15                      |
| Classes    | Parasitized, Uninfected |

### Architecture

```
MobileNetV2 (ImageNet)
        │
GlobalAveragePooling
        │
Dense (128, ReLU)
        │
Dense (1, Sigmoid)
```

### Data Augmentation

* Rotation
* Width shift
* Height shift
* Zoom
* Horizontal flip

---

# 2️⃣ Brain Tumor Detection

**Goal:** Classify MRI images into tumor types.

### Model

Xception (Transfer Learning)

### Configuration

| Parameter  | Value                                   |
| ---------- | --------------------------------------- |
| Image Size | 299 × 299                               |
| Optimizer  | Adam                                    |
| Loss       | Categorical Crossentropy                |
| Epochs     | 15                                      |
| Classes    | Glioma, Meningioma, Pituitary, No Tumor |

### Architecture

```
Xception (ImageNet)
        │
Global Max Pooling
        │
Flatten
        │
Dense (128)
        │
Dropout
        │
Dense (4, Softmax)
```

### Training Strategy

Stage 1

* Freeze base layers
* Train classifier

Stage 2

* Unfreeze model
* Fine tune with smaller learning rate

---

# 3️⃣ Skin Disease Detection

**Goal:** Classify dermatological diseases from skin images.

### Model

Deep Learning Ensemble Model

### Models Used

* EfficientNet-B4
* ConvNeXt-Base
* Vision Transformer (ViT-B16)

### Input

```
Image Size: 224 × 224
Classes: 22 Skin Diseases
```

### Ensemble Strategy

```
Final Prediction =
0.6 × ConvNeXt +
0.4 × Vision Transformer
```

This combination improves:

* Model generalization
* Classification accuracy
* Robustness across skin tones

---

# 📊 Datasets Used

## Skin Disease Dataset

Dermatological image dataset containing **22 skin disease categories**.

## Brain Tumor Dataset

MRI images categorized into:

* Glioma
* Meningioma
* Pituitary tumor
* No tumor

## Malaria Dataset

Microscopic blood smear images categorized into:

* Parasitized
* Uninfected

---

# 🛠 Tech Stack

## Backend

* Python
* FastAPI
* TensorFlow / PyTorch
* NumPy
* OpenCV

## Frontend

* React
* JavaScript
* HTML5
* CSS

## AI / Machine Learning

* Convolutional Neural Networks
* Transfer Learning
* Vision Transformers
* Ensemble Learning

---

# 📂 Project Structure

```
Mediscan
│
├── backend
│   ├── main.py
│   ├── routes
│   ├── models
│   └── utils
│
├── frontend
│   ├── src
│   ├── components
│   └── pages
│
├── notebooks
│   ├── malaria_detection.ipynb
│   ├── brain_tumor_detection.ipynb
│   └── skin_disease_detection.ipynb
│
├── datasets
│
└── README.md
```

---

# ⚙ Installation

## Clone the Repository

```bash
git clone https://github.com/PatilVaishnav131/disease-prediction-ai.git
cd disease-prediction-ai
```

---

# Backend Setup

```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Server will run at

```
http://localhost:8000
```

API Docs available at

```
http://localhost:8000/docs
```

---

# Frontend Setup

```
cd frontend
npm install
npm start
```

Frontend runs at

```
http://localhost:3000
```

---

# 🔌 API Endpoints

## Skin Disease Prediction

```
POST /predict/skin
```

Upload skin image to receive predicted disease.

---

## Brain Tumor Detection

```
POST /predict/brain
```

Upload MRI image for tumor classification.

---

## Malaria Detection

```
POST /predict/malaria
```

Upload blood smear image for malaria detection.

---

# 💡 Future Improvements

* Add **more disease detection models**
* Integrate **patient health record system**
* Add **doctor consultation module**
* Deploy using **Docker + Cloud**
* Add **explainable AI (Grad-CAM visualization)**

---

# ⚠ Disclaimer

This project is intended for **educational and research purposes only**.
It should **not replace professional medical diagnosis or treatment**.

---

# 👨‍💻 Contributors

* **Sarthak Patil**
* **Vaishnav Patil**
* **Paras Patil**
* **Nikhil Pawar**

