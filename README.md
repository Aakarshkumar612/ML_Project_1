## End To End Machine_Learning_Project ##

MLFLOW_TRACKING_URI=https://dagshub.com/Aakarshkumar612/ML_Project_1.mlflow \
MLFLOW_TRACKING_USERNAME=Aakarshkumar612 \
MLFLOW_TRACKING_PASSWORD=b2660137767ee17015a9c1f699a0f4161e50d91c \
python script.py

---

## 🚀 Streamlit Web Application

This project includes a professional, modern Streamlit frontend for interactive model predictions and analysis.

### Features

- **📁 CSV Upload**: Upload CSV files with student data for batch predictions
- **✏️ Manual Input**: Enter individual student data using interactive sliders and dropdowns
- **📊 Visualizations**: Interactive charts including prediction distributions, residual plots, and feature importance
- **📈 Model Metrics**: Real-time calculation of R², MAE, MSE, and RMSE when target values are provided
- **🔍 Model Explanation**: Feature importance analysis to understand model decisions
- **📥 Export Results**: Download predictions as CSV files
- **🎨 Modern UI**: Clean, responsive design with dark navy theme and teal accents

### Running Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model Artifacts Exist**:
   Make sure you have trained the model and that the following files exist:
   - `artifacts/model.pkl`
   - `artifacts/preprocessor.pkl`

   If not, train the model first:
   ```bash
   python app.py
   ```

3. **Launch Streamlit App**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the App**:
   Open your browser to `http://localhost:8501`

### Testing with Demo Data

A sample CSV file is provided in `assets/demo_data.csv` for testing the application. You can upload this file to see predictions and model metrics.

### Deployment to Streamlit Cloud

1. **Push to GitHub**:
   Ensure your repository is pushed to GitHub (https://github.com/Aakarshkumar612/ML_Project_1)

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository: `Aakarshkumar612/ML_Project_1`
   - Set the main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Important Notes for Streamlit Cloud**:
   - Ensure `artifacts/model.pkl` and `artifacts/preprocessor.pkl` are committed to the repository
   - The app will automatically install dependencies from `requirements.txt`
   - If your model files are large, consider using Git LFS or hosting them externally

### Docker Deployment

1. **Create a Dockerfile** (if not already present):
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
   
   ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build the Docker Image**:
   ```bash
   docker build -t ml-project-streamlit .
   ```

3. **Run the Container**:
   ```bash
   docker run -p 8501:8501 ml-project-streamlit
   ```

4. **Access the App**:
   Open your browser to `http://localhost:8501`

### Project Structure

```
ML_Project/
├── streamlit_app.py          # Main Streamlit application
├── src/
│   └── ui/
│       └── helpers.py        # Helper functions for model loading and predictions
├── assets/
│   └── demo_data.csv         # Sample data for testing
├── artifacts/
│   ├── model.pkl            # Trained model (must exist)
│   └── preprocessor.pkl     # Preprocessing pipeline (must exist)
└── requirements.txt          # Python dependencies
```

### Usage Guide

#### CSV Upload Mode:
1. Click on the "📁 Upload CSV" tab
2. Upload a CSV file with the required columns:
   - `writing_score` (numerical, 0-100)
   - `reading_score` (numerical, 0-100)
   - `gender` (categorical: "male" or "female")
   - `race_ethnicity` (categorical: "group A", "group B", "group C", "group D", or "group E")
   - `parental_level_of_education` (categorical)
   - `lunch` (categorical: "standard" or "free/reduced")
   - `test_preparation_course` (categorical: "none" or "completed")
3. Optionally select a target column (`math_score`) for model evaluation
4. Click "🔮 Predict" to generate predictions
5. View results, charts, and download predictions as CSV

#### Manual Input Mode:
1. Click on the "✏️ Manual Input" tab
2. Adjust sliders and select dropdown values for each feature
3. Click "🔮 Predict" to get a single prediction
4. View the predicted math score

#### Model Explanation:
- Click the "🔍 Explain Model" button to view feature importances
- This helps understand which features most influence predictions

### Troubleshooting

- **Model not found error**: Ensure `artifacts/model.pkl` and `artifacts/preprocessor.pkl` exist. Train the model using `python app.py` if needed.
- **Missing columns error**: Verify your CSV contains all required feature columns (see Usage Guide above).
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`

### Requirements

See `requirements.txt` for the complete list of dependencies. Key packages include:
- `streamlit>=1.28.0`
- `plotly>=5.17.0`
- `pandas`
- `numpy`
- `scikit-learn`
- `catboost` or `xgboost` (depending on trained model)