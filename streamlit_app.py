"""
Professional Streamlit Frontend for ML Project
A modern, responsive UI for student performance prediction model.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import io

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from src.ui.helpers import (
    load_model_and_preprocessor,
    validate_input_data,
    make_predictions,
    calculate_metrics,
    get_feature_importances,
    get_model_info,
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    ALL_FEATURE_COLUMNS,
)

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f1724 0%, #1e293b 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: #06b6d4;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-header p {
        color: #cbd5e1;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #06b6d4;
    }
    
    .stButton>button {
        background-color: #06b6d4;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #0891b2;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(6, 182, 212, 0.3);
    }
    
    .stSelectbox label, .stSlider label, .stTextInput label {
        font-weight: 500;
        color: #1e293b;
    }
    
    .info-box {
        background: #f1f5f9;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #06b6d4;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #fee2e2;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #ef4444;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d1fae5;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'manual_inputs' not in st.session_state:
    st.session_state.manual_inputs = {}

# Load model (cached)
@st.cache_resource
def get_model():
    try:
        return load_model_and_preprocessor()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure model.pkl and preprocessor.pkl exist in the artifacts/ directory.")
        return None, None

# Hero Header
st.markdown("""
<div class="main-header">
    <h1>📊 Student Performance Predictor</h1>
    <p>Predict math scores using machine learning. Upload a CSV or enter data manually to get predictions with model explanations.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for advanced settings
with st.sidebar:
    st.header("⚙️ Advanced Settings")
    
    with st.expander("Model Configuration", expanded=False):
        model_version = st.selectbox(
            "Model Version",
            ["v1.0 (Current)", "v0.9 (Legacy)"],
            help="Select the model version to use for predictions"
        )
        
        scaling_method = st.selectbox(
            "Scaling Method",
            ["StandardScaler (Default)", "MinMaxScaler", "RobustScaler"],
            help="Preprocessing scaling method (model uses StandardScaler)"
        )
        
        imputation_method = st.selectbox(
            "Imputation Method",
            ["Median (Numerical) / Mode (Categorical)", "Mean (Numerical) / Mode (Categorical)"],
            help="Method for handling missing values"
        )
        
        random_seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=9999,
            value=42,
            help="Random seed for reproducibility"
        )
    
    st.markdown("---")
    
    # Model Card
    st.header("📋 Model Card")
    model_info = get_model_info()
    st.write(f"**Model Type:** {model_info['name']}")
    st.write(f"**Has Feature Importances:** {'Yes' if model_info['has_feature_importances'] else 'No'}")
    
    with st.expander("Model Details"):
        st.json(model_info.get('parameters', {}))
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.85rem; color: #64748b;">
        <p><strong>About:</strong></p>
        <p>This model predicts student math scores based on demographic and academic features.</p>
        <p><strong>Features:</strong></p>
        <ul>
            <li>Writing Score</li>
            <li>Reading Score</li>
            <li>Gender</li>
            <li>Race/Ethnicity</li>
            <li>Parental Education</li>
            <li>Lunch Type</li>
            <li>Test Preparation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main content area
tab1, tab2 = st.tabs(["📁 Upload CSV", "✏️ Manual Input"])

# Tab 1: CSV Upload
with tab1:
    st.header("Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with student data. Required columns: writing_score, reading_score, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Show preview
            with st.expander("📋 Data Preview (First 5 rows)", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
            
            # Column mapping
            st.subheader("Column Mapping")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Feature Columns:**")
                feature_cols = st.multiselect(
                    "Select feature columns",
                    options=df.columns.tolist(),
                    default=[col for col in ALL_FEATURE_COLUMNS if col in df.columns],
                    help="Select all columns that will be used as features"
                )
            
            with col2:
                st.write("**Target Column (Optional):**")
                target_col = st.selectbox(
                    "Select target column for evaluation",
                    options=["None"] + df.columns.tolist(),
                    help="If your CSV contains actual math scores, select the column for model evaluation"
                )
                if target_col == "None":
                    target_col = None
            
            # Validate and process
            if st.button("🔮 Predict", type="primary", use_container_width=True):
                with st.spinner("Processing data and making predictions..."):
                    try:
                        # Validate columns
                        missing_cols = [col for col in ALL_FEATURE_COLUMNS if col not in df.columns]
                        if missing_cols:
                            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                        else:
                            # Load model
                            model, preprocessor = get_model()
                            if model is None or preprocessor is None:
                                st.error("Model could not be loaded. Please check the artifacts directory.")
                            else:
                                # Prepare data
                                input_df = df[ALL_FEATURE_COLUMNS].copy()
                                
                                # Make predictions
                                predictions = make_predictions(model, preprocessor, input_df)
                                
                                # Store in session state
                                st.session_state.predictions = predictions
                                st.session_state.input_data = df.copy()
                                
                                # Calculate metrics if target column provided
                                if target_col and target_col in df.columns:
                                    actual = df[target_col].values
                                    metrics = calculate_metrics(actual, predictions)
                                    st.session_state.metrics = metrics
                                    
                                    st.success("✅ Predictions generated successfully!")
                                    st.balloons()
                                else:
                                    st.session_state.metrics = None
                                    st.success("✅ Predictions generated successfully!")
                                
                                st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")

# Tab 2: Manual Input
with tab2:
    st.header("Enter Data Manually")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numerical Features")
        writing_score = st.slider(
            "Writing Score",
            min_value=0,
            max_value=100,
            value=70,
            help="Student's writing score (0-100)"
        )
        reading_score = st.slider(
            "Reading Score",
            min_value=0,
            max_value=100,
            value=70,
            help="Student's reading score (0-100)"
        )
    
    with col2:
        st.subheader("Categorical Features")
        gender = st.selectbox(
            "Gender",
            options=["male", "female"],
            help="Student's gender"
        )
        race_ethnicity = st.selectbox(
            "Race/Ethnicity",
            options=["group A", "group B", "group C", "group D", "group E"],
            help="Student's race/ethnicity group"
        )
        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            options=["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"],
            help="Highest level of education completed by parents"
        )
        lunch = st.selectbox(
            "Lunch Type",
            options=["standard", "free/reduced"],
            help="Type of lunch program"
        )
        test_preparation_course = st.selectbox(
            "Test Preparation Course",
            options=["none", "completed"],
            help="Whether student completed test preparation course"
        )
    
    if st.button("🔮 Predict", type="primary", use_container_width=True, key="manual_predict"):
        with st.spinner("Making prediction..."):
            try:
                # Create DataFrame from inputs
                input_data = pd.DataFrame([{
                    "writing_score": writing_score,
                    "reading_score": reading_score,
                    "gender": gender,
                    "race_ethnicity": race_ethnicity,
                    "parental_level_of_education": parental_level_of_education,
                    "lunch": lunch,
                    "test_preparation_course": test_preparation_course,
                }])
                
                # Load model
                model, preprocessor = get_model()
                if model is None or preprocessor is None:
                    st.error("Model could not be loaded. Please check the artifacts directory.")
                else:
                    # Make prediction
                    prediction = make_predictions(model, preprocessor, input_data)
                    
                    # Store in session state
                    st.session_state.predictions = prediction
                    st.session_state.input_data = input_data
                    st.session_state.metrics = None
                    
                    st.success("✅ Prediction generated successfully!")
                    st.balloons()
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Results Section
if st.session_state.predictions is not None:
    st.markdown("---")
    st.header("📊 Results")
    
    # Model Summary Card
    col1, col2, col3, col4 = st.columns(4)
    
    model_info = get_model_info()
    with col1:
        st.metric("Model", model_info['name'])
    
    if st.session_state.metrics:
        with col2:
            st.metric("R² Score", f"{st.session_state.metrics['r2']:.4f}")
        with col3:
            st.metric("MAE", f"{st.session_state.metrics['mae']:.2f}")
        with col4:
            st.metric("RMSE", f"{st.session_state.metrics['rmse']:.2f}")
    else:
        with col2:
            st.metric("Predictions", len(st.session_state.predictions))
    
    # Create results DataFrame
    results_df = st.session_state.input_data.copy()
    results_df['predicted_math_score'] = st.session_state.predictions
    
    if st.session_state.metrics and TARGET_COLUMN in results_df.columns:
        results_df['actual_math_score'] = results_df[TARGET_COLUMN]
        results_df['residual'] = results_df['actual_math_score'] - results_df['predicted_math_score']
    
    # Charts Section
    if len(st.session_state.predictions) > 1:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("📈 Predictions Distribution")
            fig_hist = px.histogram(
                results_df,
                x='predicted_math_score',
                nbins=30,
                title="Distribution of Predicted Math Scores",
                labels={'predicted_math_score': 'Predicted Math Score', 'count': 'Frequency'}
            )
            fig_hist.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter", size=12)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with chart_col2:
            if st.session_state.metrics and 'actual_math_score' in results_df.columns:
                st.subheader("🎯 Predicted vs Actual")
                fig_scatter = px.scatter(
                    results_df,
                    x='actual_math_score',
                    y='predicted_math_score',
                    title="Predicted vs Actual Math Scores",
                    labels={'actual_math_score': 'Actual Math Score', 'predicted_math_score': 'Predicted Math Score'},
                    trendline="ols"
                )
                fig_scatter.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Inter", size=12)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.subheader("📊 Prediction Summary")
                st.dataframe(
                    results_df[ALL_FEATURE_COLUMNS + ['predicted_math_score']].head(20),
                    use_container_width=True
                )
        
        # Residual Plot
        if st.session_state.metrics and 'residual' in results_df.columns:
            st.subheader("📉 Residual Plot")
            fig_residual = px.scatter(
                results_df,
                x='predicted_math_score',
                y='residual',
                title="Residual Plot",
                labels={'predicted_math_score': 'Predicted Math Score', 'residual': 'Residual (Actual - Predicted)'},
                hover_data=ALL_FEATURE_COLUMNS[:3]
            )
            fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
            fig_residual.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter", size=12)
            )
            st.plotly_chart(fig_residual, use_container_width=True)
    
    # Feature Importance
    if st.button("🔍 Explain Model", type="secondary"):
        with st.spinner("Calculating feature importances..."):
            try:
                model, preprocessor = get_model()
                if model is None:
                    st.error("Model not loaded")
                else:
                    importances = get_feature_importances(model, preprocessor)
                    if importances:
                        # Sort by importance
                        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
                        
                        # Create bar chart
                        fig_importance = px.bar(
                            x=list(sorted_importances.values()),
                            y=list(sorted_importances.keys()),
                            orientation='h',
                            title="Feature Importance",
                            labels={'x': 'Importance', 'y': 'Feature'},
                            color=list(sorted_importances.values()),
                            color_continuous_scale='cyan'
                        )
                        fig_importance.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            font=dict(family="Inter", size=12),
                            height=400
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.info("Feature importances not available for this model type.")
            except Exception as e:
                st.error(f"Error calculating feature importances: {str(e)}")
    
    # Results Table
    st.subheader("📋 Results Table (Top 20)")
    display_cols = ALL_FEATURE_COLUMNS + ['predicted_math_score']
    if 'actual_math_score' in results_df.columns:
        display_cols.append('actual_math_score')
    if 'residual' in results_df.columns:
        display_cols.append('residual')
    
    st.dataframe(
        results_df[display_cols].head(20),
        use_container_width=True,
        height=400
    )
    
    # Download button
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_buffer.getvalue(),
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.predictions = None
            st.session_state.input_data = None
            st.session_state.metrics = None
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <p>Built with ❤️ using Streamlit | Student Performance Prediction Model</p>
    <p style="font-size: 0.85rem;">For issues or questions, please check the repository documentation.</p>
</div>
""", unsafe_allow_html=True)

