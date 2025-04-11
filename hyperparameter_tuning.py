import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import streamlit as st

def get_param_grid(model_name):
    """
    Define hyperparameter search spaces for different models.
    
    Parameters:
        model_name (str): Name of the model
    
    Returns:
        dict: Hyperparameter search space
    """
    if model_name == 'Logistic Regression':
        param_grid = {
            'C': np.logspace(-4, 4, 20),
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 500, 1000, 2000]
        }
        # Handle solver/penalty combinations that are not allowed
        param_grid['solver_penalty'] = [
            ('newton-cg', 'l2'), ('newton-cg', 'none'),
            ('lbfgs', 'l2'), ('lbfgs', 'none'),
            ('liblinear', 'l1'), ('liblinear', 'l2'),
            ('sag', 'l2'), ('sag', 'none'),
            ('saga', 'l1'), ('saga', 'l2'), ('saga', 'elasticnet'), ('saga', 'none')
        ]
    
    elif model_name == 'Decision Tree':
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    
    elif model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
    
    elif model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
    
    elif model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'reg_alpha': [0, 0.1, 1, 10],
            'reg_lambda': [0, 0.1, 1, 10]
        }
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return param_grid

def get_base_model(model_name):
    """
    Get a new instance of the base model for hyperparameter tuning.
    
    Parameters:
        model_name (str): Name of the model
    
    Returns:
        object: Model instance
    """
    if model_name == 'Logistic Regression':
        return LogisticRegression(random_state=42)
    elif model_name == 'Decision Tree':
        return DecisionTreeClassifier(random_state=42)
    elif model_name == 'Random Forest':
        return RandomForestClassifier(random_state=42)
    elif model_name == 'Gradient Boosting':
        return GradientBoostingClassifier(random_state=42)
    elif model_name == 'XGBoost':
        return XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError(f"Unknown model: {model_name}")

def perform_hyperparameter_tuning(model_name, X_train, y_train, n_iter=10, cv=3, scoring='accuracy'):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    Parameters:
        model_name (str): Name of the model to tune
        X_train (array-like): Training feature data
        y_train (array-like): Training target data
        n_iter (int): Number of parameter settings to sample
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric for model evaluation
    
    Returns:
        tuple: (best_model, best_params, cv_results)
    """
    # Get parameter grid
    param_grid = get_param_grid(model_name)
    
    # Get base model
    base_model = get_base_model(model_name)
    
    # Special handling for Logistic Regression due to solver/penalty constraints
    if model_name == 'Logistic Regression':
        # Extract solver_penalty combinations
        solver_penalty_combinations = param_grid.pop('solver_penalty')
        
        # Create a list to store all valid parameter combinations
        valid_param_combinations = []
        
        # For each valid solver-penalty combination
        for solver, penalty in solver_penalty_combinations:
            # Create a valid param dict
            valid_params = {
                'solver': [solver],
                'penalty': [penalty],
                'C': param_grid['C'],
                'max_iter': param_grid['max_iter']
            }
            
            # For 'none' penalty, C doesn't matter
            if penalty == 'none':
                valid_params['C'] = [1.0]  # Default value
            
            # For elasticnet, only saga solver works and l1_ratio is needed
            if penalty == 'elasticnet':
                valid_params['l1_ratio'] = np.linspace(0, 1, 10)
            
            valid_param_combinations.append(valid_params)
        
        # Create RandomizedSearchCV for each valid combination and find the best
        best_score = -np.inf
        best_model = None
        best_params = None
        all_cv_results = []
        
        for params in valid_param_combinations:
            search = RandomizedSearchCV(
                base_model,
                param_distributions=params,
                n_iter=max(1, n_iter // len(valid_param_combinations)),
                cv=cv,
                scoring=scoring,
                random_state=42,
                n_jobs=-1,
                return_train_score=True
            )
            
            # Fit the search
            search.fit(X_train, y_train)
            
            # Save CV results
            results = pd.DataFrame(search.cv_results_)
            all_cv_results.append(results)
            
            # Check if this is the best model so far
            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_model = search.best_estimator_
                best_params = search.best_params_
        
        # Combine all CV results
        cv_results = pd.concat(all_cv_results, ignore_index=True)
        
    else:
        # For other models, use standard RandomizedSearchCV
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )
        
        # Fit the search
        search.fit(X_train, y_train)
        
        # Get best model and parameters
        best_model = search.best_estimator_
        best_params = search.best_params_
        cv_results = pd.DataFrame(search.cv_results_)
    
    return best_model, best_params, cv_results
