import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import streamlit as st

def load_data(file):
    """Load data from CSV file or URL"""
    try:
        if isinstance(file, str) and (file.startswith('http://') or file.startswith('https://')):
            # Load from URL
            df = pd.read_csv(file)
        else:
            # Load from uploaded file
            df = pd.read_csv(file)
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) > 1 else None)

def prepare_data(df, target_column, categorical_columns, numerical_columns, 
                 columns_to_drop=None, handle_missing=True, test_size=0.2, random_state=42):
    """
    Prepare data for modeling by handling missing values, encoding categorical features,
    scaling numerical features, and splitting into train and test sets.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Drop specified columns
    if columns_to_drop:
        data = data.drop(columns=columns_to_drop)
    
    # Extract target variable
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    # Check if target is categorical and encode if needed
    if data[target_column].dtype == 'object' or pd.api.types.is_categorical_dtype(data[target_column]):
        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])
        target_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        st.write("Target encoding mapping:", target_mapping)
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Verify categorical and numerical columns are in X
    categorical_columns = [col for col in categorical_columns if col in X.columns]
    numerical_columns = [col for col in numerical_columns if col in X.columns]
    
    # Check if we have at least one column to process
    if not (categorical_columns or numerical_columns):
        raise ValueError("No valid categorical or numerical columns specified.")
    
    # Define preprocessing for numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median' if handle_missing else 'constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent' if handle_missing else 'constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessors into a single transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after transformation
    feature_names = []
    if numerical_columns:
        feature_names.extend(numerical_columns)
    
    if categorical_columns:
        # Get transformed categorical feature names
        cat_features = []
        for i, col in enumerate(categorical_columns):
            categories = preprocessor.transformers_[1][1].named_steps['onehot'].categories_[i]
            for cat in categories:
                cat_features.append(f"{col}_{cat}")
        feature_names.extend(cat_features)
    
    # Split the processed data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_names
