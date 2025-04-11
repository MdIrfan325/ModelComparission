"""
This module contains predefined optimal hyperparameters for various machine learning models
based on common classification tasks. These can serve as good starting points for model training.
"""

import numpy as np

def get_optimal_hyperparameters(model_name):
    """
    Get optimal hyperparameters for a given model.
    
    Parameters:
        model_name (str): Name of the model
    
    Returns:
        dict: Dictionary of optimal hyperparameters
    """
    if model_name == 'Logistic Regression':
        return {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 1000,
            'class_weight': 'balanced'
        }
    
    elif model_name == 'Decision Tree':
        return {
            'criterion': 'gini',
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'class_weight': 'balanced'
        }
    
    elif model_name == 'Random Forest':
        return {
            'n_estimators': 200,
            'criterion': 'gini',
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'class_weight': 'balanced',
            'oob_score': True
        }
    
    elif model_name == 'Gradient Boosting':
        return {
            'n_estimators': 150,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'subsample': 0.8,
            'max_features': 'sqrt'
        }
    
    elif model_name == 'XGBoost':
        return {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': 1
        }
    
    else:
        return {}

def use_optimal_hyperparameters(model, model_name):
    """
    Set optimal hyperparameters for a given model instance.
    
    Parameters:
        model: The model instance to set parameters for
        model_name (str): Name of the model
    
    Returns:
        object: Model with optimal hyperparameters
    """
    optimal_params = get_optimal_hyperparameters(model_name)
    
    if optimal_params:
        # Set parameters for the model
        for param, value in optimal_params.items():
            if hasattr(model, param):
                setattr(model, param, value)
    
    return model