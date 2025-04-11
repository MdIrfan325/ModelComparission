import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import streamlit as st
from utils import set_matplotlib_style

# Set the matplotlib style for better visualizations
COLORS = set_matplotlib_style()

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
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # If there's only one model, axes won't be an array
    if n_models == 1:
        axes = np.array([axes])
    # Flatten axes array for easier iteration
    axes = axes.flatten()
    
    for i, (model_name, results) in enumerate(evaluation_results.items()):
        cm = results['confusion_matrix']
        
        # Plot confusion matrix with enhanced styling
        cmap = sns.color_palette("Blues", as_cmap=True)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap=cmap, 
            ax=axes[i],
            cbar=True,
            linewidths=1,
            linecolor='white',
            square=True,
            annot_kws={"size": 12, "weight": "bold"},
        )
        
        # Improve title and label styling
        axes[i].set_title(f'{model_name}\nConfusion Matrix', 
                         fontsize=14, 
                         fontweight='bold', 
                         pad=15)
        axes[i].set_xlabel('Predicted Label', 
                          fontsize=12, 
                          fontweight='semibold', 
                          labelpad=10)
        axes[i].set_ylabel('True Label', 
                          fontsize=12, 
                          fontweight='semibold', 
                          labelpad=10)
        
        # Add box around the plot
        for spine in axes[i].spines.values():
            spine.set_visible(True)
            spine.set_color('#dddddd')
            spine.set_linewidth(1)
    
    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout(pad=3.0)
    return fig

def plot_roc_curves(evaluation_results):
    """
    Create a figure with ROC curves for each model.
    
    Parameters:
        evaluation_results (dict): Dictionary of model evaluation results
    
    Returns:
        matplotlib.figure.Figure: Figure containing ROC curves
    """
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Check if ROC curve data is available
    has_roc_data = False
    
    # Use cycle of line styles and widths for better distinction
    line_styles = ['-', '--', '-.', ':']
    line_widths = [2.5, 2.5, 2.5, 2.5]
    
    for i, (model_name, results) in enumerate(evaluation_results.items()):
        if results['roc_curve_data'] is not None:
            has_roc_data = True
            fpr = results['roc_curve_data']['fpr']
            tpr = results['roc_curve_data']['tpr']
            roc_auc = results['roc_curve_data']['roc_auc']
            
            # Get color, line style, and width for this model
            color = COLORS[i % len(COLORS)]
            ls = line_styles[i % len(line_styles)]
            lw = line_widths[i % len(line_widths)]
            
            # Plot with enhanced styling
            ax.plot(
                fpr, tpr, 
                lw=lw, 
                ls=ls,
                color=color,
                alpha=0.8,
                label=f'{model_name} (AUC = {roc_auc:.3f})'
            )
            
            # Add light fill under the curve for visual appeal (with low alpha for overlap)
            ax.fill_between(fpr, tpr, alpha=0.1, color=color)
    
    if has_roc_data:
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Guess')
        
        # Grid, limits, and labels with improved styling
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        
        # Add shaded regions to indicate performance quality zones
        ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.1, color='green', label='Good Performance')
        ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.1, color='red', label='Poor Performance')
        
        # Enhanced axis titles and overall title
        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='semibold', labelpad=10)
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='semibold', labelpad=10)
        ax.set_title('Receiver Operating Characteristic (ROC) Curves', 
                    fontsize=16, fontweight='bold', pad=15)
        
        # Enhanced legend
        legend = ax.legend(
            loc="lower right", 
            fontsize=11, 
            frameon=True, 
            fancybox=True, 
            framealpha=0.9,
            shadow=True,
            borderpad=1
        )
        
        # Add spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#dddddd')
            spine.set_linewidth(1)
            
    else:
        # Improved message for non-binary classification
        ax.text(0.5, 0.5, 
               'ROC curves are only available for binary classification problems', 
               fontsize=14,
               color='#666666',
               fontweight='semibold',
               horizontalalignment='center', 
               verticalalignment='center')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.axis('off')
    
    plt.tight_layout()
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
        
        # Create figure with enhanced styling
        fig, ax = plt.subplots(figsize=(12, max(8, min(16, top_n * 0.4))), facecolor='white')
        fig.patch.set_facecolor('white')
        
        # Generate a gradient color palette for the bars
        palette = sns.color_palette("Blues_r", n_colors=len(feature_importance))
        
        # Plot horizontal bar chart with enhanced styling
        bars = sns.barplot(
            x='Importance', 
            y='Feature', 
            data=feature_importance, 
            palette=palette,
            ax=ax
        )
        
        # Add value labels at the end of each bar
        for i, p in enumerate(bars.patches):
            width = p.get_width()
            ax.text(
                width + 0.005, 
                p.get_y() + p.get_height()/2, 
                f'{width:.4f}',
                ha='left',
                va='center',
                fontweight='bold',
                fontsize=10,
                color='#444444'
            )
        
        # Add a subtle grid for better readability
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Enhanced title and labels
        ax.set_title(f'Top {top_n} Most Important Features - {model_type}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Relative Importance', fontsize=13, fontweight='semibold', labelpad=10)
        ax.set_ylabel('Feature Name', fontsize=13, fontweight='semibold', labelpad=10)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Style remaining spines
        ax.spines['left'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#dddddd')
        
        # Add a subtitle explaining feature importance
        model_explanation = ""
        if model_type == 'Logistic Regression':
            model_explanation = "Feature importance based on absolute coefficient values"
        elif model_type in ['Random Forest', 'Decision Tree', 'Gradient Boosting', 'XGBoost']:
            model_explanation = "Feature importance based on information gain/Gini impurity reduction"
            
        if model_explanation:
            plt.figtext(
                0.5, 0.01, 
                model_explanation, 
                ha='center', 
                fontsize=11, 
                fontstyle='italic',
                color='#666666'
            )
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig
    
    except Exception as e:
        # If feature importance can't be calculated, return a message with enhanced styling
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
        fig.patch.set_facecolor('white')
        
        # Create a styled error message
        error_text = f"Feature importance not available for this model:\n{str(e)}"
        
        # Add a visually appealing error message box
        error_box = plt.Rectangle((0.1, 0.3), 0.8, 0.4, fill=True, color='#f8d7da', 
                                 alpha=0.6, transform=ax.transAxes)
        ax.add_patch(error_box)
        
        ax.text(0.5, 0.5, error_text, 
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=12,
                color='#721c24',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=1', facecolor='#f8d7da', 
                         alpha=0.1, edgecolor='#f5c6cb', linewidth=2))
        
        ax.set_title('Feature Importance Error', fontsize=14, pad=20, color='#721c24')
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
    
    # Create figure with enhanced styling
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), facecolor='white')
    fig.patch.set_facecolor('white')
    
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
            
            # Enhanced line plot for numeric parameters
            color = COLORS[i % len(COLORS)]
            axes[i].plot(
                sorted_results[param_col], 
                sorted_results['mean_test_score'], 
                marker='o', 
                markersize=8,
                markerfacecolor=color,
                markeredgewidth=1.5,
                markeredgecolor='white',
                linestyle='-', 
                linewidth=2.5,
                color=color,
                alpha=0.8
            )
            
            # Add shaded confidence interval if std_test_score is available
            if 'std_test_score' in sorted_results.columns:
                axes[i].fill_between(
                    sorted_results[param_col],
                    sorted_results['mean_test_score'] - sorted_results['std_test_score'],
                    sorted_results['mean_test_score'] + sorted_results['std_test_score'],
                    alpha=0.2,
                    color=color
                )
            
            # Find and mark the best parameter value
            best_idx = sorted_results['mean_test_score'].idxmax()
            best_param = sorted_results.loc[best_idx, param_col]
            best_score = sorted_results.loc[best_idx, 'mean_test_score']
            
            axes[i].scatter(
                [best_param], [best_score],
                s=150,
                c='red',
                marker='*',
                edgecolor='white',
                linewidth=1.5,
                zorder=10,
                label=f'Best: {best_param}'
            )
            
            # Enhanced styling
            axes[i].grid(True, linestyle='--', alpha=0.3)
            axes[i].set_xlabel(param, fontsize=12, fontweight='semibold', labelpad=10)
            axes[i].set_ylabel('Mean CV Score', fontsize=12, fontweight='semibold', labelpad=10)
            axes[i].set_title(f'Impact of {param}', fontsize=14, fontweight='bold', pad=15)
            axes[i].legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True)
            
        else:
            # For categorical parameters - enhanced bar plot
            param_scores = cv_results.groupby(param_col)['mean_test_score'].mean().reset_index()
            
            # Add error bars if std_test_score is available
            if 'std_test_score' in cv_results.columns:
                param_std = cv_results.groupby(param_col)['std_test_score'].mean().reset_index()
                param_scores = pd.merge(param_scores, param_std, on=param_col)
                
                # Sort by performance
                param_scores = param_scores.sort_values('mean_test_score', ascending=False)
                
                # Get best parameter value
                best_param = param_scores.iloc[0][param_col]
                
                # Create palette with best parameter highlighted
                n_bars = len(param_scores)
                colors = [COLORS[0]] * n_bars
                for j in range(n_bars):
                    if param_scores.iloc[j][param_col] == best_param:
                        colors[j] = '#EF4444'  # Highlight color for best
                
                # Plot with error bars
                sns.barplot(
                    x=param_col, 
                    y='mean_test_score', 
                    data=param_scores, 
                    ax=axes[i],
                    palette=colors,
                    edgecolor='white',
                    linewidth=1.5
                )
                
                # Add error bars
                for j, (_, row) in enumerate(param_scores.iterrows()):
                    axes[i].errorbar(
                        j, row['mean_test_score'], 
                        yerr=row['std_test_score'],
                        fmt='none', 
                        color='black', 
                        capsize=5, 
                        capthick=1.5,
                        elinewidth=1.5,
                        alpha=0.7
                    )
            else:
                # Simple bar plot without error bars
                param_scores = param_scores.sort_values('mean_test_score', ascending=False)
                sns.barplot(
                    x=param_col, 
                    y='mean_test_score', 
                    data=param_scores, 
                    ax=axes[i],
                    palette=[COLORS[i % len(COLORS)]] * len(param_scores)
                )
            
            # Add value labels on top of bars
            for j, p in enumerate(axes[i].patches):
                axes[i].annotate(
                    f"{p.get_height():.4f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center',
                    va = 'bottom',
                    fontsize=9,
                    fontweight='bold',
                    color='#444444',
                    xytext=(0, 5),
                    textcoords='offset points'
                )
            
            # Enhanced styling
            axes[i].grid(axis='y', linestyle='--', alpha=0.3)
            axes[i].set_xlabel(param, fontsize=12, fontweight='semibold', labelpad=10)
            axes[i].set_ylabel('Mean CV Score', fontsize=12, fontweight='semibold', labelpad=10)
            axes[i].set_title(f'Impact of {param}', fontsize=14, fontweight='bold', pad=15)
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
        
        # Style spines
        for spine in axes[i].spines.values():
            spine.set_color('#dddddd')
    
    # If no varying parameters, show a styled message
    if n_params == 0:
        axes[0].text(0.5, 0.5, 
                    'No varying parameters found in hyperparameter search', 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    fontsize=14,
                    color='#666666',
                    fontweight='semibold',
                    transform=axes[0].transAxes,
                    bbox=dict(boxstyle='round,pad=1', 
                             facecolor='#f0f0f0', 
                             alpha=0.5, 
                             edgecolor='#dddddd'))
        axes[0].axis('off')
    
    # Hide any unused subplots
    for j in range(n_params, len(axes)):
        axes[j].axis('off')
    
    # Add a descriptive figure title
    title_text = f'Hyperparameter Tuning Results for {model_name}'
    subtitle_text = 'Higher scores indicate better model performance on validation data'
    
    plt.suptitle(title_text, fontsize=18, fontweight='bold', y=0.98)
    plt.figtext(0.5, 0.91, subtitle_text, fontsize=12, ha='center', color='#666666')
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    return fig
