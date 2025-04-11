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
    /* Main background with a more appealing color theme */
    .main {
        background-color: #f8f9fc;
    }
    .stApp {
        background: linear-gradient(135deg, #f8f9fc 0%, #eef2ff 50%, #e0e7ff 100%);
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    
    /* Vibrant heading styles with gradient effects */
    h1, h2, h3 {
        color: #4338ca;
        font-weight: 600;
    }
    h1 {
        background: linear-gradient(45deg, #4338ca, #6366f1, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        padding: 10px 0;
        text-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
    }
    h2 {
        color: #4f46e5;
        border-bottom: 2px solid #e0e7ff;
        padding-bottom: 8px;
    }
    h3 {
        color: #4f46e5;
    }
    
    /* Sidebar with a more readable color scheme */
    [data-testid="stSidebar"] {
        background: linear-gradient(175deg, #4338ca, #6366f1);
        border-right: 1px solid rgba(99, 102, 241, 0.3);
        padding: 1rem 0;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding-top: 0.5rem;
        gap: 0.5rem;
    }
    
    /* Sidebar title and heading styles */
    [data-testid="stSidebar"] .css-6qob1r {
        color: white !important;
    }
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Fix for navigation text */
    [data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"] {
        color: white !important;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        margin-bottom: 0.2rem;
        padding: 0.5rem !important;
    }
    
    /* Make sidebar navigation items stand out on hover */
    [data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"]:hover {
        background-color: rgba(255, 255, 255, 0.2);
        transition: all 0.2s ease;
    }
    
    /* Sidebar headers - make them stand out */
    [data-testid="stSidebar"] h3 {
        color: white !important;
        font-weight: 600;
        padding: 15px 15px 10px 15px;
        margin-top: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.3);
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Ensure dropdowns and selections have visible text */
    [data-testid="stSidebar"] select option {
        color: #111827 !important;
        background-color: white;
    }
    
    /* Style for select option in sidebar */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div > div {
        color: white !important;
        font-weight: 500;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    /* Dropdown background color */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 8px 10px;
        margin: 5px 0 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Make the arrows in dropdowns visible */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
        color: white !important;
    }
    
    /* Make buttons more vibrant and interactive */
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5, #3b82f6);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(59, 130, 246, 0.4);
        background: linear-gradient(90deg, #4338ca, #3b82f6);
    }
    [data-testid="stSidebar"] .stButton > button {
        background: white;
        color: #3b82f6;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15);
        background: #f9fafb;
    }
    
    /* Improve dataframe appearance */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #dbeafe;
        box-shadow: 0 4px 6px rgba(219, 234, 254, 0.5);
    }
    .stDataFrame th {
        background: linear-gradient(90deg, #4f46e5, #3b82f6) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px 12px !important;
    }
    .stDataFrame td {
        padding: 10px 12px !important;
    }
    .stDataFrame tr:nth-child(even) {
        background-color: #f8fafc !important;
    }
    .stDataFrame tr:hover {
        background-color: #eff6ff !important;
    }
    
    /* Text and paragraph styles */
    p, li, div {
        color: #1e293b;
    }
    a {
        color: #4f46e5;
        text-decoration: none;
        font-weight: 500;
    }
    a:hover {
        text-decoration: underline;
    }
    
    /* Colorful information boxes */
    .info-box {
        background: linear-gradient(to right, rgba(59, 130, 246, 0.05), rgba(59, 130, 246, 0.1));
        border-left: 5px solid #3b82f6;
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(219, 234, 254, 0.5);
    }
    .success-box {
        background: linear-gradient(to right, rgba(34, 197, 94, 0.05), rgba(34, 197, 94, 0.1));
        border-left: 5px solid #22c55e;
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(220, 252, 231, 0.5);
    }
    .warning-box {
        background: linear-gradient(to right, rgba(245, 158, 11, 0.05), rgba(245, 158, 11, 0.1));
        border-left: 5px solid #f59e0b;
        padding: 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(254, 243, 199, 0.5);
    }
    
    /* Enhance tabs appearance */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #dbeafe;
    }
    .stTabs [role="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px;
        border: 1px solid #dbeafe;
        border-bottom: none;
        color: #64748b;
        transition: all 0.2s ease;
    }
    .stTabs [role="tab"]:hover {
        background-color: #f8fafc;
        color: #3b82f6;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(to bottom, #eff6ff, white) !important;
        color: #1e40af !important;
        font-weight: 600 !important;
        border-top: 3px solid #3b82f6 !important;
    }
    
    /* Section dividers */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #3b82f6, transparent);
        border-radius: 2px;
        margin: 2rem 0;
    }
    
    /* Make expanders more attractive */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease;
    }
    .streamlit-expanderHeader:hover {
        background-color: #eff6ff;
    }
    .streamlit-expanderContent {
        background-color: white;
        border-radius: 0 0 8px 8px;
        border-top: none;
        padding: 1rem;
        border: 1px solid #dbeafe;
        border-top: none;
    }
    
    /* Metric styling with more colorful appearance */
    [data-testid="stMetric"] {
        background: linear-gradient(to right, rgba(59, 130, 246, 0.05), rgba(59, 130, 246, 0.02));
        border-radius: 8px;
        padding: 10px 15px !important;
        border: 1px solid rgba(219, 234, 254, 0.5);
    }
    [data-testid="stMetric"] label {
        color: #3b82f6 !important;
        font-weight: 600 !important;
    }
    
    /* Improve radio button appearance */
    .stRadio [role="radiogroup"] {
        padding: 10px;
        border-radius: 8px;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .stRadio [role="radio"] {
        border-color: #3b82f6 !important;
    }
    
    /* Checkbox styling */
    .stCheckbox label {
        font-weight: 500;
        color: #334155;
    }
    .stCheckbox [data-baseweb="checkbox"] div:first-child {
        background-color: #eff6ff !important;
        border-color: #3b82f6 !important;
    }
    
    /* Create badge-like styles for status */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-left: 0.5rem;
    }
    .badge-blue {
        background-color: #dbeafe;
        color: #1e40af;
    }
    .badge-green {
        background-color: #dcfce7;
        color: #166534;
    }
    .badge-yellow {
        background-color: #fef3c7;
        color: #92400e;
    }
    .badge-red {
        background-color: #fee2e2;
        color: #b91c1c;
    }
    .badge-purple {
        background-color: #ede9fe;
        color: #5b21b6;
    }
    </style>
    """, unsafe_allow_html=True)

def set_matplotlib_style():
    """Set the Matplotlib style for better visualizations"""
    plt.style.use('ggplot')
    
    # Enhanced color palette - more vibrant and professional
    colors = [
        '#4f46e5',  # Indigo
        '#f59e0b',  # Amber
        '#10b981',  # Emerald
        '#ef4444',  # Red
        '#8b5cf6',  # Purple
        '#ec4899',  # Pink
        '#06b6d4',  # Cyan
        '#84cc16',  # Lime
        '#f97316',  # Orange
        '#7c3aed',  # Violet
        '#0ea5e9',  # Sky
        '#14b8a6'   # Teal
    ]
    sns.set_palette(sns.color_palette(colors))
    
    # Enhanced figure aesthetics
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 120  # Higher DPI for sharper graphics
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    
    # Title and label styling
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.labelweight'] = 'semibold'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    # Legend styling
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.shadow'] = True
    plt.rcParams['legend.edgecolor'] = '#d1d5db'
    
    # Grid and background styling
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.color'] = '#d1d5db'
    plt.rcParams['axes.facecolor'] = '#f8fafc'  # Lighter background
    plt.rcParams['figure.facecolor'] = '#ffffff'
    
    # Bar and line styling
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['patch.edgecolor'] = 'white'
    plt.rcParams['patch.linewidth'] = 1
    
    # Add more breathing room around plots
    plt.rcParams['figure.subplot.top'] = 0.92
    plt.rcParams['figure.subplot.right'] = 0.95
    plt.rcParams['figure.subplot.bottom'] = 0.12
    plt.rcParams['figure.subplot.left'] = 0.10
    
    # Spines (borders) styling
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    return colors

def display_footer():
    """Display the footer with creator information"""
    st.markdown("---")
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 50%, #60a5fa 100%); 
                    padding: 1.5rem; border-radius: 10px; margin-top: 2rem; box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);">
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="width: 70%; vertical-align: middle; padding-right: 30px;">
                        <div style="font-weight: bold; font-size: 1.3em; color: white; margin-bottom: 8px;">
                            ML Model Comparison Dashboard
                        </div>
                        <div style="font-size: 0.9em; color: rgba(255, 255, 255, 0.9); line-height: 1.4;">
                            A premium tool for comparing machine learning classification models with comprehensive metrics, 
                            visualizations, and optimization capabilities.
                        </div>
                    </td>
                    <td style="width: 30%; vertical-align: middle; border-left: 1px solid rgba(255, 255, 255, 0.3); padding-left: 30px;">
                        <div style="font-size: 0.9em; color: white; margin-bottom: 5px;">
                            <strong>Created by:</strong>
                        </div>
                        <div style="font-weight: 600; font-size: 1.1em; color: white; margin-bottom: 10px; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);">
                            Mohammed Irfan
                        </div>
                        <div style="font-size: 0.9em; margin-bottom: 8px;">
                            <a href="mailto:mi3253050@gmail.com" 
                               style="color: white; text-decoration: none; background-color: rgba(255, 255, 255, 0.2); 
                                     padding: 5px 10px; border-radius: 5px; display: inline-block;">
                                mi3253050@gmail.com
                            </a>
                        </div>
                        <div style="font-size: 0.8em; color: rgba(255, 255, 255, 0.8);">
                            © 2025 • Premium ML Visualization Tool
                        </div>
                    </td>
                </tr>
            </table>
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
