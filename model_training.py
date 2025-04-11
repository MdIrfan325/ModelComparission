import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import streamlit as st
from optimal_hyperparameters import get_optimal_hyperparameters

def train_models(X_train, y_train, models_to_train, use_optimal=True):
    """
    Train selected machine learning models.
    
    Parameters:
        X_train (array-like): Training feature data
        y_train (array-like): Training target data
        models_to_train (dict): Dictionary specifying which models to train
        use_optimal (bool): Whether to use optimal hyperparameters
    
    Returns:
        dict: Dictionary containing trained models and training times
    """
    models = {}
    
    # Train Logistic Regression
    if models_to_train.get('Logistic Regression', False):
        start_time = time.time()
        
        if use_optimal:
            # Get optimal hyperparameters
            optimal_params = get_optimal_hyperparameters('Logistic Regression')
            lr = LogisticRegression(
                C=optimal_params.get('C', 1.0),
                penalty=optimal_params.get('penalty', 'l2'),
                solver=optimal_params.get('solver', 'lbfgs'),
                max_iter=optimal_params.get('max_iter', 1000),
                class_weight=optimal_params.get('class_weight', 'balanced'),
                random_state=42
            )
        else:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            
        lr.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        models['Logistic Regression'] = {
            'model': lr,
            'train_time': train_time
        }
    
    # Train Decision Tree
    if models_to_train.get('Decision Tree', False):
        start_time = time.time()
        
        if use_optimal:
            # Get optimal hyperparameters
            optimal_params = get_optimal_hyperparameters('Decision Tree')
            dt = DecisionTreeClassifier(
                criterion=optimal_params.get('criterion', 'gini'),
                max_depth=optimal_params.get('max_depth', 8),
                min_samples_split=optimal_params.get('min_samples_split', 5),
                min_samples_leaf=optimal_params.get('min_samples_leaf', 2),
                max_features=optimal_params.get('max_features', 'sqrt'),
                class_weight=optimal_params.get('class_weight', 'balanced'),
                random_state=42
            )
        else:
            dt = DecisionTreeClassifier(random_state=42)
            
        dt.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        models['Decision Tree'] = {
            'model': dt,
            'train_time': train_time
        }
    
    # Train Random Forest
    if models_to_train.get('Random Forest', False):
        start_time = time.time()
        
        if use_optimal:
            # Get optimal hyperparameters
            optimal_params = get_optimal_hyperparameters('Random Forest')
            rf = RandomForestClassifier(
                n_estimators=optimal_params.get('n_estimators', 200),
                criterion=optimal_params.get('criterion', 'gini'),
                max_depth=optimal_params.get('max_depth', 15),
                min_samples_split=optimal_params.get('min_samples_split', 5),
                min_samples_leaf=optimal_params.get('min_samples_leaf', 2),
                max_features=optimal_params.get('max_features', 'sqrt'),
                bootstrap=optimal_params.get('bootstrap', True),
                class_weight=optimal_params.get('class_weight', 'balanced'),
                oob_score=optimal_params.get('oob_score', True),
                random_state=42,
                n_jobs=-1
            )
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
        rf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        models['Random Forest'] = {
            'model': rf,
            'train_time': train_time
        }
    
    # Train Gradient Boosting
    if models_to_train.get('Gradient Boosting', False):
        start_time = time.time()
        
        if use_optimal:
            # Get optimal hyperparameters
            optimal_params = get_optimal_hyperparameters('Gradient Boosting')
            gb = GradientBoostingClassifier(
                n_estimators=optimal_params.get('n_estimators', 150),
                learning_rate=optimal_params.get('learning_rate', 0.1),
                max_depth=optimal_params.get('max_depth', 5),
                min_samples_split=optimal_params.get('min_samples_split', 5),
                min_samples_leaf=optimal_params.get('min_samples_leaf', 2),
                subsample=optimal_params.get('subsample', 0.8),
                max_features=optimal_params.get('max_features', 'sqrt'),
                random_state=42
            )
        else:
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            
        gb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        models['Gradient Boosting'] = {
            'model': gb,
            'train_time': train_time
        }
    
    # Train XGBoost
    if models_to_train.get('XGBoost', False):
        start_time = time.time()
        
        if use_optimal:
            # Get optimal hyperparameters
            optimal_params = get_optimal_hyperparameters('XGBoost')
            xgb = XGBClassifier(
                n_estimators=optimal_params.get('n_estimators', 200),
                learning_rate=optimal_params.get('learning_rate', 0.1),
                max_depth=optimal_params.get('max_depth', 6),
                min_child_weight=optimal_params.get('min_child_weight', 3),
                subsample=optimal_params.get('subsample', 0.8),
                colsample_bytree=optimal_params.get('colsample_bytree', 0.8),
                gamma=optimal_params.get('gamma', 0.1),
                reg_alpha=optimal_params.get('reg_alpha', 0.1),
                reg_lambda=optimal_params.get('reg_lambda', 1.0),
                scale_pos_weight=optimal_params.get('scale_pos_weight', 1),
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
            
        xgb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        models['XGBoost'] = {
            'model': xgb,
            'train_time': train_time
        }
    
    return models
