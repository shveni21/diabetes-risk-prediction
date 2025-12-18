# Fingerprint-Based Diabetes Risk Prediction using Random Forest

This project presents a **non-invasive machine learning approach to assess diabetes risk levels** using fingerprint-derived dermatoglyphic features. The system **does not diagnose diabetes**; instead, it classifies individuals into **Low, Medium, or High risk categories**, supporting early screening and preventive awareness.  
The core model is a **Random Forest classifier achieving ~95% accuracy** on the evaluated dataset.

---

## 🔍 Problem Overview
Early identification of individuals at risk of diabetes enables timely lifestyle and clinical interventions. This project explores fingerprint-based features as a **screening-level, non-diagnostic alternative** to conventional methods and benchmarks performance against traditional risk-factor–based approaches.

---

## 🧠 Approach
- Extracted handcrafted dermatoglyphic features such as ridge density, minutiae count, pattern class, and ridge spacing
- Trained and optimized a **Random Forest classifier** for **risk-level prediction (Low / Medium / High)**
- Evaluated model performance using standard classification metrics
- Benchmarked results against baseline risk models based on BMI, age, and family history

---

## 📊 Results
- **Model:** Random Forest Classifier  
- **Output:** Diabetes risk level (Low / Medium / High)  
- **Accuracy:** ~95%  
- **Learning Type:** Supervised Learning  
- **Evaluation Metrics:** Accuracy, Precision, Recall, Confusion Matrix  

The model demonstrated strong predictive performance for **risk stratification**, making it suitable for population-level screening use cases.

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Machine Learning:** Scikit-learn, NumPy, Pandas  
- **Model Inference:** Simple Flask application  
- **Data Analysis:** Jupyter Notebook  

---

## 📂 Project Structure
```
│── train.py # Model training and evaluation
│── app.py # Lightweight Flask-based inference
│── fingerprint_dataset.csv # Dataset
│── requirements.txt
│
├── notebooks/ # Data analysis and experimentation
├── model/ # Saved trained models
└── static / templates # Minimal UI for testing predictions
```

---

## ▶️ Running the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Train the model
```bash
python train.py
```
### 3. Run inference (optional)
```bash
python app.py
```


