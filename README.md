# 🩺 MediPredict — AI-Powered Disease Prediction & Health Guidance Platform

MediPredict is a modern full-stack medical web application that helps the general public and doctors to assess potential disease risks through AI/ML models and learn about health with expert-backed guidance.

---

## 🌐 Tech Stack

### 🖥️ Frontend
- React.js (with React Router)
- TailwindCSS (fully responsive UI)
- Axios for API integration

### 🧠 Backend
- Django + Django REST Framework
- SQLite / PostgreSQL (configurable)
- Python ML Model (scikit-learn, pandas, etc.)

---

## 🤖 Machine Learning Integration

Trained ML models (like Logistic Regression, Random Forest, etc.) are integrated to analyze user health reports and symptoms to predict diseases such as:
- Heart Disease
- Diabetes
- Lung cancer

Model is either:
- Loaded directly inside Django
- OR served separately via Flask/FastAPI (microservice)

---

## 📦 Features

- 🧬 **Disease Prediction** based on uploaded reports or questionnaire
- 📊 **Interactive Frontend** with real-time prediction results
- 📚 **Health Education Section** (Tips, FAQs, Guides)
- 📂 **User History Tracking** (WIP)
- 🔐 **Secure Authentication** (coming soon)
- ⚙️ **Admin Panel** (Django built-in)

---

## 📁 Project Structure

```bash
disease_predictions/
├── frontend/                # ReactJS codebase
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   ├── pages/
│   │   └── App.js
│   └── public/
├── backend/                 # Django backend
│   ├── medi_backend/
│   ├── disease_model/
│   └── manage.py
├── ML_models/               # Trained ML models (.pkl or .joblib)
└── README.md
