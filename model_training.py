import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import streamlit as st

def train_models(X_train, y_train, models_to_train):
    """
    Train selected machine learning models.
    
    Parameters:
        X_train (array-like): Training feature data
        y_train (array-like): Training target data
        models_to_train (dict): Dictionary specifying which models to train
    
    Returns:
        dict: Dictionary containing trained models and training times
    """
    models = {}
    
    # Train Logistic Regression
    if models_to_train.get('Logistic Regression', False):
        start_time = time.time()
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
        xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        models['XGBoost'] = {
            'model': xgb,
            'train_time': train_time
        }
    
    return models
