import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import streamlit as st
from utils import calculate_metrics

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test data.
    
    Parameters:
        models (dict): Dictionary of trained models
        X_test (array-like): Test feature data
        y_test (array-like): Test target data
    
    Returns:
        dict: Dictionary containing evaluation results for each model
    """
    results = {}
    
    # Check if binary or multiclass
    n_classes = len(np.unique(y_test))
    is_binary = n_classes == 2
    
    for model_name, model_info in models.items():
        model = model_info['model'] if isinstance(model_info, dict) else model_info
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability estimates if possible
        try:
            y_proba = model.predict_proba(X_test)
            if is_binary:
                y_prob = y_proba[:, 1]  # For binary classification, use proba of positive class
            else:
                y_prob = y_proba  # For multiclass, keep the full probability matrix
        except:
            y_prob = None
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Get classification report
        report = classification_report(y_test, y_pred)
        
        # Calculate ROC curve and AUC for binary classification
        if is_binary and y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            roc_curve_data = {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
        else:
            roc_curve_data = None
        
        # Store results
        results[model_name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'confusion_matrix': cm,
            'classification_report': report,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'roc_curve_data': roc_curve_data
        }
    
    return results
