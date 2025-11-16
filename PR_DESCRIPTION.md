# Streamlit Frontend Implementation

## Summary

This PR adds a professional, modern Streamlit web application frontend for the ML Project, enabling interactive model predictions through both CSV upload and manual input modes. The UI features a clean, responsive design with dark navy theme and teal accents, comprehensive visualizations, model explanation capabilities, and export functionality.

## Changes

### New Files
- **`streamlit_app.py`**: Main Streamlit application with full UI implementation
  - Two-column layout (inputs on left, outputs on right)
  - CSV upload mode with column mapping
  - Manual input mode with sliders and dropdowns
  - Interactive visualizations (prediction distributions, residual plots, feature importance)
  - Model metrics calculation (R², MAE, MSE, RMSE)
  - Export predictions as CSV
  - Collapsible sidebar with advanced settings and model card

- **`src/ui/helpers.py`**: Helper module for model loading and preprocessing
  - Cached model/preprocessor loading
  - Data validation and preprocessing wrappers
  - Prediction functions
  - Metrics calculation
  - Feature importance extraction

- **`src/ui/__init__.py`**: Package initialization

- **`assets/demo_data.csv`**: Sample dataset for testing (20 rows)

### Modified Files
- **`requirements.txt`**: Added Streamlit and visualization dependencies
  - `streamlit>=1.28.0`
  - `plotly>=5.17.0`
  - `shap>=0.42.0`
  - `streamlit-lottie>=0.0.5`

- **`README.md`**: Added comprehensive Streamlit app documentation
  - Local running instructions
  - Streamlit Cloud deployment guide
  - Docker deployment instructions
  - Usage guide for both input modes
  - Troubleshooting section

## Features

### UI/UX
- Modern, responsive design with custom CSS styling
- Dark navy header (#0f1724) with teal/cyan accents (#06b6d4)
- Two-column desktop layout, mobile-responsive
- Progress spinners and success animations
- Tooltips and helpful placeholder text
- Accessible keyboard-focusable controls

### Functionality
- **CSV Upload**: Batch predictions with optional target column for evaluation
- **Manual Input**: Single prediction with interactive controls
- **Visualizations**: 
  - Prediction distribution histograms
  - Predicted vs Actual scatter plots (when target available)
  - Residual plots
  - Feature importance bar charts
- **Model Metrics**: Real-time R², MAE, MSE, RMSE calculation
- **Export**: Download predictions as CSV
- **Model Card**: Sidebar with model metadata and parameters

### Technical
- Cached model loading using `@st.cache_resource`
- Error handling with user-friendly messages
- Session state management for predictions and results
- Integration with existing model artifacts (`artifacts/model.pkl`, `artifacts/preprocessor.pkl`)

## How to Run

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure model artifacts exist (train if needed)
python app.py

# Launch Streamlit app
streamlit run streamlit_app.py
```

### Streamlit Cloud
1. Push to GitHub
2. Deploy at share.streamlit.io
3. Select repository and set main file to `streamlit_app.py`

### Docker
```bash
docker build -t ml-project-streamlit .
docker run -p 8501:8501 ml-project-streamlit
```

## Testing

Use the provided `assets/demo_data.csv` file to test the application. The demo file contains 20 sample student records with all required features and target values.

## Notes

- Model artifacts (`artifacts/model.pkl` and `artifacts/preprocessor.pkl`) must exist before running the app
- The app automatically handles missing values using the saved preprocessor
- Feature importance extraction works for tree-based models (Random Forest, XGBoost, CatBoost, etc.)
- The UI is fully responsive and works on both desktop and mobile devices

