# 🏦 Customer Churn Prediction using ANN

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow.svg)

## 📌 Project Overview
Customer attrition (churn) is one of the biggest challenges for banks and financial institutions. This end-to-end machine learning project leverages an Artificial Neural Network (ANN) to predict the likelihood of a customer leaving the bank based on their demographic and financial profile. 

By identifying at-risk customers early, businesses can proactively apply retention strategies.

### 🚀 Live Demo
**Try the interactive web application here:** https://customer-churn-using-ann-jbxappygvcqvnycebsqpkfe.streamlit.app/

*(Note: The app may take a few seconds to wake up if it hasn't been used recently.)*

---

## 🧠 What I Learned
Building this end-to-end data science project solidified several core concepts in deep learning and model deployment:

* **Deep Learning Architecture:** Designing, compiling, and training a Sequential Artificial Neural Network (ANN) using TensorFlow and Keras.
* **Data Preprocessing:** Handling categorical variables using `OneHotEncoder` and `LabelEncoder`, and understanding the critical importance of feature scaling (`StandardScaler`) for gradient descent-based models.
* **Model Serialization:** Saving and loading trained models (`.h5` format) and preprocessing pipelines (`.pkl` files) to ensure consistent predictions in production.
* **Web App Deployment:** Translating a Jupyter Notebook backend into a fully interactive, user-friendly frontend using Streamlit.
* **Dependency Management:** Navigating and resolving real-world version conflicts for successful cloud deployment.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Data Manipulation & Math:** Pandas, NumPy
* **Machine Learning & Preprocessing:** Scikit-Learn
* **Frontend/UI:** Streamlit, Custom CSS

---

## 💻 How to Run Locally

If you want to clone this repository and run the application on your own machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/MadniSheikh/Customer-Churn-Using-ANN.git
cd Customer-Churn-Using-ANN
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows use:
venv\Scripts\activate
# On Mac/Linux use:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 📂 Repository Structure
```text
├── Data/
├── encoders/
├── Model/
├── Notebook/
│   ├── logs/
│   ├── Experiments.ipynb
│   ├── prediction.ipynb
│   └── Training.ipynb
├── .gitignore
├── app.py
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🤝 Connect with Me

Hi, I'm Madni! I am passionate about bridging the gap between Artificial Intelligence and Financial Services, building end-to-end machine learning systems that solve real-world problems. Let's connect:

* **LinkedIn:** www.linkedin.com/in/mohmedmadni-sheikh-
* **GitHub:** https://github.com/MadniSheikh
* **Kaggle:** https://www.kaggle.com/madnishaikh
