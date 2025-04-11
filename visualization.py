import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import streamlit as st

def plot_model_comparison(evaluation_results):
    """
    Create a DataFrame comparing model metrics.
    
    Parameters:
        evaluation_results (dict): Dictionary of model evaluation results
    
    Returns:
        pandas.DataFrame: DataFrame containing model comparison metrics
    """
    metrics = []
    for model_name, results in evaluation_results.items():
        metrics.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1 Score': results['f1'],
            'ROC AUC': results['roc_auc'] if results['roc_auc'] is not None else np.nan
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    # Set Model as index and highlight the best value in each metric column
    metrics_df = metrics_df.set_index('Model')
    
    return metrics_df

def plot_confusion_matrices(evaluation_results):
    """
    Create a figure with confusion matrices for each model.
    
    Parameters:
        evaluation_results (dict): Dictionary of model evaluation results
    
    Returns:
        matplotlib.figure.Figure: Figure containing confusion matrices
    """
    n_models = len(evaluation_results)
    
    # Determine grid layout based on number of models
    if n_models <= 3:
        n_cols = n_models
        n_rows = 1
    elif n_models <= 6:
        n_cols = 3
        n_rows = 2
    else:
        n_cols = 3
        n_rows = (n_models + 2) // 3  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    # If there's only one model, axes won't be an array
    if n_models == 1:
        axes = np.array([axes])
    # Flatten axes array for easier iteration
    axes = axes.flatten()
    
    for i, (model_name, results) in enumerate(evaluation_results.items()):
        cm = results['confusion_matrix']
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{model_name}\nConfusion Matrix')
        axes[i].set_xlabel('Predicted label')
        axes[i].set_ylabel('True label')
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    return fig

def plot_roc_curves(evaluation_results):
    """
    Create a figure with ROC curves for each model.
    
    Parameters:
        evaluation_results (dict): Dictionary of model evaluation results
    
    Returns:
        matplotlib.figure.Figure: Figure containing ROC curves
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Check if ROC curve data is available
    has_roc_data = False
    
    for model_name, results in evaluation_results.items():
        if results['roc_curve_data'] is not None:
            has_roc_data = True
            fpr = results['roc_curve_data']['fpr']
            tpr = results['roc_curve_data']['tpr']
            roc_auc = results['roc_curve_data']['roc_auc']
            
            ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    if has_roc_data:
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, 'ROC curves are only available for binary classification problems', 
                horizontalalignment='center', verticalalignment='center')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.axis('off')
    
    return fig

def plot_feature_importance(model, feature_names, model_type=None, top_n=20):
    """
    Create a figure with feature importance for the given model.
    
    Parameters:
        model: Trained model
        feature_names (list): List of feature names
        model_type (str): Type of the model
        top_n (int): Number of top features to display
    
    Returns:
        matplotlib.figure.Figure: Figure containing feature importance plot
    """
    try:
        # Get feature importance
        if model_type in ['Logistic Regression']:
            # For logistic regression, use coefficients as importance
            importance = np.abs(model.coef_[0])
        elif hasattr(model, 'feature_importances_'):
            # For tree-based models
            importance = model.feature_importances_
        else:
            raise ValueError("Model doesn't have feature importances")
        
        # Create a dataframe for sorting and plotting
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        
        # Sort by importance and take top_n
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
        
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        return fig
    
    except Exception as e:
        # If feature importance can't be calculated, return a message
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f'Feature importance not available for this model: {str(e)}', 
                horizontalalignment='center', verticalalignment='center')
        ax.axis('off')
        return fig

def plot_hyperparameter_impact(cv_results, model_name):
    """
    Create a figure showing the impact of hyperparameters on model performance.
    
    Parameters:
        cv_results (pandas.DataFrame): DataFrame with cross-validation results
        model_name (str): Name of the model
    
    Returns:
        matplotlib.figure.Figure: Figure containing hyperparameter impact plots
    """
    # Extract the mean test scores
    cv_results = cv_results.sort_values('mean_test_score', ascending=False)
    
    # Get the parameters that vary in the search
    varying_params = []
    for param in cv_results.columns:
        if param.startswith('param_'):
            param_values = cv_results[param].nunique()
            if param_values > 1:
                varying_params.append(param.replace('param_', ''))
    
    # Limit to the top 4 varying parameters for visualization clarity
    if len(varying_params) > 4:
        varying_params = varying_params[:4]
    
    # Determine the plot layout
    n_params = len(varying_params)
    if n_params == 0:
        n_cols, n_rows = 1, 1
    elif n_params <= 2:
        n_cols, n_rows = n_params, 1
    else:
        n_cols, n_rows = 2, (n_params + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    
    # Convert to numpy array for easier indexing
    if n_params == 0:
        axes = np.array([axes])
    elif n_params == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Create a plot for each parameter
    for i, param in enumerate(varying_params):
        param_col = f'param_{param}'
        
        # Group by parameter and calculate mean test score
        if pd.api.types.is_numeric_dtype(cv_results[param_col]):
            # For numeric parameters
            # Sort the dataframe by parameter value
            sorted_results = cv_results.sort_values(param_col)
            axes[i].plot(sorted_results[param_col], sorted_results['mean_test_score'], 'o-')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Mean Test Score')
            axes[i].set_title(f'Impact of {param}')
        else:
            # For categorical parameters
            param_scores = cv_results.groupby(param_col)['mean_test_score'].mean().reset_index()
            sns.barplot(x=param_col, y='mean_test_score', data=param_scores, ax=axes[i])
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Mean Test Score')
            axes[i].set_title(f'Impact of {param}')
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    # If no varying parameters, show a message
    if n_params == 0:
        axes[0].text(0.5, 0.5, 'No varying parameters found in hyperparameter search', 
                    horizontalalignment='center', verticalalignment='center')
        axes[0].axis('off')
    
    # Hide any unused subplots
    for j in range(n_params, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f'Hyperparameter Tuning Results for {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle
    
    return fig
