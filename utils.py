import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def set_page_config():
    """Set the Streamlit page configuration"""
    st.set_page_config(
        page_title="ML Model Comparison Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def display_footer():
    """Display the footer with creator information"""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: right; font-size: 0.8em; color: #888;">
        Created by: Mohammed Irfan | Contact: mi3253050@gmail.com
        </div>
        """,
        unsafe_allow_html=True
    )

def format_time(time_in_seconds):
    """Format time in seconds to a human-readable format"""
    if time_in_seconds < 60:
        return f"{time_in_seconds:.2f} seconds"
    elif time_in_seconds < 3600:
        minutes = int(time_in_seconds // 60)
        seconds = time_in_seconds % 60
        return f"{minutes} min {seconds:.2f} sec"
    else:
        hours = int(time_in_seconds // 3600)
        minutes = int((time_in_seconds % 3600) // 60)
        seconds = time_in_seconds % 60
        return f"{hours} hr {minutes} min {seconds:.2f} sec"

def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate various classification metrics"""
    if y_prob is None:
        y_prob = y_pred  # For models that don't provide probability estimates
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    # For binary classification, calculate ROC AUC if possible
    if len(np.unique(y_true)) == 2 and y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
    
    return metrics

def create_styled_dataframe(df, highlight_col=None, highlight_max=True):
    """Create a styled dataframe with highlights"""
    styled_df = df.style
    
    if highlight_col is not None:
        # Highlight the max or min value in the specified column
        if highlight_max:
            styled_df = styled_df.highlight_max(subset=[highlight_col], color='lightgreen')
        else:
            styled_df = styled_df.highlight_min(subset=[highlight_col], color='lightcoral')
    
    return styled_df

def safe_division(numerator, denominator):
    """Safely divide two numbers, return 0 if denominator is 0"""
    return numerator / denominator if denominator != 0 else 0
