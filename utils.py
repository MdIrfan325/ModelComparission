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
    
    # Set custom theme using CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        background: linear-gradient(to bottom right, #f5f7f9, #e8eef2);
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
        font-weight: 600;
    }
    h1 {
        background: linear-gradient(45deg, #1E3A8A, #3563E9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    .stButton button {
        background: linear-gradient(90deg, #3563E9, #1E3A8A);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(29, 78, 216, 0.3);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #eaeaea;
    }
    .stDataFrame th {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    .css-1hverof, .css-15tx938 {
        font-size: 0.9rem;
    }
    .css-6qob1r {
        background-image: linear-gradient(to bottom, #1E3A8A, #3563E9);
        color: white;
    }
    .css-6qob1r p, .css-6qob1r h2 {
        color: white !important;
    }
    .css-6qob1r [data-testid="stVerticalBlock"] {
        gap: 0.25rem;
    }
    .sidebar-content {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .info-box {
        background-color: rgba(53, 99, 233, 0.1);
        border-left: 5px solid #3563E9;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
    }
    .success-box {
        background-color: rgba(34, 197, 94, 0.1);
        border-left: 5px solid #22C55E;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
    }
    .warning-box {
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def set_matplotlib_style():
    """Set the Matplotlib style for better visualizations"""
    plt.style.use('ggplot')
    
    # Custom color palette - vibrant and professional
    colors = ['#3563E9', '#F59E0B', '#22C55E', '#EF4444', '#8B5CF6', 
              '#EC4899', '#06B6D4', '#84CC16', '#F97316', '#A855F7']
    sns.set_palette(sns.color_palette(colors))
    
    # Figure aesthetics
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.shadow'] = True
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['figure.facecolor'] = '#ffffff'
    
    return colors

def display_footer():
    """Display the footer with creator information"""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: right; background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-top: 3px solid #3563E9;">
            <div style="font-weight: bold; font-size: 1.1em; color: #1E3A8A;">ML Model Comparison Dashboard</div>
            <div style="font-size: 0.9em; color: #4B5563; margin-top: 0.3rem;">
                Created by: <span style="color: #1E3A8A; font-weight: 500;">Mohammed Irfan</span> | 
                <a href="mailto:mi3253050@gmail.com" style="color: #3563E9; text-decoration: none;">mi3253050@gmail.com</a>
            </div>
            <div style="font-size: 0.8em; color: #6B7280; margin-top: 0.3rem;">
                © 2025 • Premium Machine Learning Visualization Tool
            </div>
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
