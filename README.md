# 🚀 Student Performance Predictor

This is an end-to-end Machine Learning project that predicts a student's final math score based on various factors.

## 🔴 Live Demo

You can view and test the live application here:
**[https://mlproject1-nj9bsop9rawzinzwnsjgtc.streamlit.app/](https://mlproject1-nj9bsop9rawzinzwnsjgtc.streamlit.app/)**

---

## 🛠️ Project Overview

This project uses a machine learning model to predict student performance. The entire pipeline is built with Python and includes:
* Data Ingestion
* Data Transformation and Preprocessing
* Model Training (using CatBoost, XGBoost, etc.)
* Model Evaluation
* Deployment as an interactive web app.

## 💻 Tech Stack
* **Python**
* **Pandas** for data manipulation
* **Scikit-learn** for preprocessing and modeling
* **Streamlit** for the interactive frontend
* **Streamlit Community Cloud** for deployment

## 🔧 How to Run Locally
1.  Clone the repository:
    ```bash
    git clone [https://github.com/Aakarshkumar612/ML_Project_1.git](https://github.com/Aakarshkumar612/ML_Project_1.git)
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/Scripts/activate
    ```
3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```