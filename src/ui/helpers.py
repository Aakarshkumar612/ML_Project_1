"""
Helper functions for Streamlit UI to load models and make predictions.
"""
import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import streamlit as st

# Add project root to path to import project modules
# helpers.py is at src/ui/helpers.py, so we need to go up 2 levels
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.ml_project.utils import load_object
from src.ml_project.exception import CustomException
from src.ml_project.logger import logging

# Feature columns expected by the model
NUMERICAL_COLUMNS = ["writing_score", "reading_score"]
CATEGORICAL_COLUMNS = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]
TARGET_COLUMN = "math_score"
ALL_FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

# Model and preprocessor paths
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")


@st.cache_resource
def load_model_and_preprocessor():
    """
    Load the trained model and preprocessor from artifacts.
    Uses Streamlit caching to avoid reloading on every rerun.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        if not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
        
        model = load_object(MODEL_PATH)
        preprocessor = load_object(PREPROCESSOR_PATH)
        
        logging.info("Model and preprocessor loaded successfully")
        return model, preprocessor
    except Exception as e:
        logging.error(f"Error loading model/preprocessor: {str(e)}")
        raise CustomException(e, sys)


def validate_input_data(df: pd.DataFrame, require_target: bool = False) -> Tuple[bool, str]:
    """
    Validate that the input DataFrame has all required columns.
    
    Args:
        df: Input DataFrame
        require_target: Whether target column is required
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_cols = []
    
    # Check feature columns
    for col in ALL_FEATURE_COLUMNS:
        if col not in df.columns:
            missing_cols.append(col)
    
    # Check target column if required
    if require_target and TARGET_COLUMN not in df.columns:
        missing_cols.append(TARGET_COLUMN)
    
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    return True, ""


def preprocess_data(df: pd.DataFrame, preprocessor) -> np.ndarray:
    """
    Preprocess input data using the saved preprocessor.
    
    Args:
        df: Input DataFrame with feature columns
        preprocessor: Fitted preprocessor object
        
    Returns:
        Preprocessed array
    """
    try:
        # Ensure columns are in correct order
        feature_df = df[ALL_FEATURE_COLUMNS].copy()
        
        # Transform using preprocessor
        processed_array = preprocessor.transform(feature_df)
        
        return processed_array
    except Exception as e:
        logging.error(f"Error preprocessing data: {str(e)}")
        raise CustomException(e, sys)


def make_predictions(model, preprocessor, df: pd.DataFrame) -> np.ndarray:
    """
    Make predictions on input data.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        df: Input DataFrame with feature columns
        
    Returns:
        Array of predictions
    """
    try:
        # Preprocess data
        processed_data = preprocess_data(df, preprocessor)
        
        # Make predictions
        predictions = model.predict(processed_data)
        
        return predictions
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        raise CustomException(e, sys)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary with metrics (r2, mae, mse, rmse)
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        "r2": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }


def get_feature_importances(model, preprocessor=None) -> Optional[Dict[str, float]]:
    """
    Extract feature importances from the model.
    
    Args:
        model: Trained model
        preprocessor: Optional preprocessor (to avoid reloading)
        
    Returns:
        Dictionary mapping feature names to importance scores, or None if not available
    """
    try:
        if hasattr(model, 'feature_importances_'):
            # Get feature names from preprocessor
            if preprocessor is None:
                preprocessor = load_object(PREPROCESSOR_PATH)
            
            # Get feature names after transformation
            # This is a simplified approach - in practice, you'd need to track
            # the feature names through the ColumnTransformer
            feature_names = []
            
            # Numerical features
            feature_names.extend(NUMERICAL_COLUMNS)
            
            # Categorical features (after one-hot encoding)
            if hasattr(preprocessor, 'named_transformers_'):
                cat_transformer = preprocessor.named_transformers_['cat_pipeline']
                if hasattr(cat_transformer, 'named_steps'):
                    ohe = cat_transformer.named_steps.get('one_hot_encoder')
                    if ohe is not None and hasattr(ohe, 'get_feature_names_out'):
                        cat_features = ohe.get_feature_names_out(CATEGORICAL_COLUMNS)
                        feature_names.extend(cat_features)
            
            # If we couldn't get proper names, use generic names
            if len(feature_names) != len(model.feature_importances_):
                feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
            
            importances = dict(zip(feature_names, model.feature_importances_))
            return importances
    except Exception as e:
        logging.warning(f"Could not extract feature importances: {str(e)}")
    
    return None


def get_model_info() -> Dict[str, Any]:
    """
    Get model metadata information.
    
    Returns:
        Dictionary with model information
    """
    try:
        model, _ = load_model_and_preprocessor()
        
        model_type = type(model).__name__
        model_name = model_type.replace("Regressor", "").replace("Classifier", "")
        
        # Get model parameters if available
        params = {}
        if hasattr(model, 'get_params'):
            params = model.get_params()
        
        return {
            "name": model_name,
            "type": model_type,
            "parameters": params,
            "has_feature_importances": hasattr(model, 'feature_importances_'),
        }
    except Exception as e:
        logging.error(f"Error getting model info: {str(e)}")
        return {
            "name": "Unknown",
            "type": "Unknown",
            "parameters": {},
            "has_feature_importances": False,
        }

