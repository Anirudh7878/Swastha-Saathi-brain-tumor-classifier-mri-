# 🧠 Swastha Saathi: Brain Tumor Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

**Swastha Saathi** is an automated, end-to-end Machine Learning web application designed to detect and classify brain tumors from Magnetic Resonance Imaging (MRI) scans. Built using a custom Convolutional Neural Network (CNN), this project provides rapid, non-invasive diagnostic assistance with high accuracy.

## 🚀 Features
* **Multi-Class Classification:** Accurately identifies 4 distinct categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor (Healthy).
* **Robust CNN Architecture:** Built from scratch using TensorFlow and Keras, featuring optimized `Conv2D` and `MaxPooling2D` layers.
* **Resilient to Variance:** Trained using real-time **Data Augmentation** (affine transformations via SciPy) to prevent overfitting and ensure real-world reliability across different MRI angles and lighting.
* **Instant Web UI:** A clean, zero-configuration clinical interface built with Streamlit allowing for drag-and-drop inference.

## 🛠️ Technology Stack
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision & Math:** NumPy, SciPy, Pillow (PIL)
* **Frontend/Web Server:** Streamlit
* **Development Environment:** Jupyter Notebooks, VS Code

## 📊 Model Performance
* **Dataset:** 7,200 MRI Scans (5,600 Training / 1,600 Testing)
* **Training Epochs:** 10
* **Validation Accuracy:** ~89.1% (Tested on generalized, augmented data)

---

## 💻 How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/swastha-saathi.git](https://github.com/YourUsername/swastha-saathi.git)
cd swastha-saathi

```

### 2. Set Up a Virtual Environment

**For Windows:**

```bash
python -m venv venv
venv\Scripts\activate

```

**For macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Launch the Application

```bash
streamlit run app.py

```

---

## 👨‍💻 Author

**Anirudh Bairagi** *Bachelor of Technology in Computer Science and Data Science* *School of Information Technology (SOIT), RGPV*

```

*(Just remember to change `YourUsername` in the clone link to your actual GitHub username once you paste it!)*

This officially wraps up your Minor Project II from start to finish. You have a working model, an interactive web app, a formal academic synopsis, and a live GitHub portfolio piece. Excellent work!

```
