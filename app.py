import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import joblib
import os
from io import StringIO

from data_preparation import prepare_data, load_data, split_data
from model_training import train_models
from model_evaluation import evaluate_models
from hyperparameter_tuning import perform_hyperparameter_tuning
from visualization import (
    plot_model_comparison,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_feature_importance,
    plot_hyperparameter_impact
)
from utils import set_page_config, display_footer

# Set page config
set_page_config()

def main():
    # Sidebar
    st.sidebar.image("https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/logos/scikit-learn-logo.png", width=200)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                            ["Home", "Data Preparation", "Model Training", 
                             "Model Evaluation", "Hyperparameter Tuning", 
                             "Comparison Dashboard"])

    # Initialize session state for data and models
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = {}
    if 'hp_results' not in st.session_state:
        st.session_state.hp_results = {}
    if 'tuned_models' not in st.session_state:
        st.session_state.tuned_models = {}
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'categorical_columns' not in st.session_state:
        st.session_state.categorical_columns = None
    if 'numerical_columns' not in st.session_state:
        st.session_state.numerical_columns = None
    if 'target_column' not in st.session_state:
        st.session_state.target_column = None
    
    # Page content
    if page == "Home":
        render_home_page()
    elif page == "Data Preparation":
        render_data_preparation_page()
    elif page == "Model Training":
        render_model_training_page()
    elif page == "Model Evaluation":
        render_model_evaluation_page()
    elif page == "Hyperparameter Tuning":
        render_hyperparameter_tuning_page()
    elif page == "Comparison Dashboard":
        render_comparison_dashboard()
    
    # Footer
    display_footer()

def render_home_page():
    st.title("🧠 Interactive ML Model Comparison Dashboard")
    
    st.markdown("""
    ### Welcome to the Premium Machine Learning Model Comparison Platform
    
    This interactive dashboard allows you to:
    - 📊 Prepare and analyze your classification dataset
    - 🔬 Train multiple machine learning models
    - 📈 Evaluate and compare model performance
    - 🛠️ Perform hyperparameter tuning to optimize models
    - 📊 Visualize results with interactive charts and graphs
    
    ### Get Started
    1. Navigate to the **Data Preparation** section to upload or use sample data
    2. Proceed to **Model Training** to train various classification models
    3. View detailed metrics in the **Model Evaluation** section
    4. Optimize models in the **Hyperparameter Tuning** section
    5. Compare all results in the **Comparison Dashboard**
    
    ### Included Models
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - XGBoost
    
    ### Available Metrics
    - Accuracy, Precision, Recall, F1-Score
    - ROC Curves and AUC
    - Confusion Matrices
    - Feature Importance
    """)
    
    st.info("👈 Use the sidebar to navigate through different sections of the application.")
    
    # Display sample images of what the dashboard can produce
    st.subheader("Example Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png", 
                 caption="Model Comparison Visualization")
    with col2:
        st.image("https://scikit-learn.org/stable/_images/sphx_glr_plot_roc_001.png", 
                 caption="ROC Curve Analysis")

def render_data_preparation_page():
    st.title("Data Preparation")
    
    st.write("Upload your dataset or use the sample Telco Customer Churn dataset.")
    
    data_option = st.radio(
        "Choose data source:",
        ("Upload CSV file", "Use sample Telco Customer Churn dataset")
    )
    
    if data_option == "Upload CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = load_data(uploaded_file)
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return
        else:
            st.info("Please upload a CSV file.")
            return
    else:
        try:
            # Try to load the sample dataset from GitHub
            url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
            data = pd.read_csv(url)
            st.success("Sample data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
            return
    
    if data is not None:
        st.session_state.data = data
        
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        st.subheader("Data Information")
        buffer = StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.subheader("Statistical Summary")
        st.dataframe(data.describe())
        
        st.subheader("Data Preparation")
        
        # Select target column
        target_column = st.selectbox("Select target column:", data.columns.tolist())
        st.session_state.target_column = target_column
        
        # Handle categorical and numerical columns
        columns_to_drop = st.multiselect("Select columns to drop (e.g., ID columns):", 
                                        data.columns.tolist(), 
                                        default=[col for col in data.columns if col.lower() in ['id', 'customerid', 'customer_id']])
        
        categorical_columns = st.multiselect(
            "Select categorical columns:",
            [col for col in data.columns if col not in columns_to_drop and col != target_column and data[col].dtype == 'object']
        )
        st.session_state.categorical_columns = categorical_columns
        
        numerical_columns = st.multiselect(
            "Select numerical columns:",
            [col for col in data.columns if col not in columns_to_drop and col != target_column and data[col].dtype != 'object']
        )
        st.session_state.numerical_columns = numerical_columns
        
        # Handle missing values
        handle_missing = st.checkbox("Handle missing values", value=True)
        
        # Test size
        test_size = st.slider("Test set size (%):", 10, 50, 20) / 100
        
        if st.button("Prepare Data"):
            with st.spinner("Preparing data..."):
                try:
                    X_train, X_test, y_train, y_test, feature_names = prepare_data(
                        data,
                        target_column,
                        categorical_columns,
                        numerical_columns,
                        columns_to_drop,
                        handle_missing,
                        test_size
                    )
                    
                    # Save to session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.feature_names = feature_names
                    
                    st.success("Data preparation completed successfully!")
                    
                    # Display shapes
                    st.write(f"Training set shape: {X_train.shape}")
                    st.write(f"Testing set shape: {X_test.shape}")
                    
                    # Class distribution
                    st.subheader("Class Distribution")
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    y_train.value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax[0], title='Training Set')
                    y_test.value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax[1], title='Testing Set')
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error during data preparation: {e}")

def render_model_training_page():
    st.title("Model Training")
    
    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning("Please prepare your data first in the Data Preparation section.")
        return
    
    st.write("Select the models you want to train:")
    
    # Model selection
    use_lr = st.checkbox("Logistic Regression", value=True)
    use_dt = st.checkbox("Decision Tree", value=True)
    use_rf = st.checkbox("Random Forest", value=True)
    use_gb = st.checkbox("Gradient Boosting", value=True)
    use_xgb = st.checkbox("XGBoost", value=True)
    
    # Optimal hyperparameters option
    use_optimal = st.checkbox("Use optimal hyperparameters", value=True, 
                             help="Use pre-defined optimal hyperparameters for each model instead of default values")
    
    if use_optimal:
        st.info("Using optimal hyperparameters for the selected models. These are research-based parameters that often work well for classification tasks.")
    
    # Training button
    if st.button("Train Models"):
        if not any([use_lr, use_dt, use_rf, use_gb, use_xgb]):
            st.error("Please select at least one model to train.")
            return
        
        with st.spinner("Training models... This may take a few moments."):
            try:
                models_to_train = {
                    'Logistic Regression': use_lr,
                    'Decision Tree': use_dt,
                    'Random Forest': use_rf,
                    'Gradient Boosting': use_gb,
                    'XGBoost': use_xgb
                }
                
                # Train models
                models = train_models(
                    st.session_state.X_train, 
                    st.session_state.y_train,
                    models_to_train,
                    use_optimal
                )
                
                # Save models to session state
                st.session_state.models = models
                
                # Evaluate models
                evaluation_results = evaluate_models(
                    models, 
                    st.session_state.X_test, 
                    st.session_state.y_test
                )
                
                # Save evaluation results to session state
                st.session_state.evaluation_results = evaluation_results
                
                st.success("Models trained successfully!")
                
                # Display training time
                for model_name, model_info in models.items():
                    st.write(f"{model_name} - Training time: {model_info['train_time']:.4f} seconds")
                
            except Exception as e:
                st.error(f"Error during model training: {e}")

def render_model_evaluation_page():
    st.title("Model Evaluation")
    
    if not st.session_state.models:
        st.warning("Please train your models first in the Model Training section.")
        return
    
    st.write("Evaluation results for trained models:")
    
    # Overall metrics comparison
    st.subheader("Performance Metrics Comparison")
    metrics_df = plot_model_comparison(st.session_state.evaluation_results)
    st.dataframe(metrics_df)
    
    # Select model for detailed evaluation
    model_names = list(st.session_state.models.keys())
    selected_model = st.selectbox("Select a model for detailed evaluation:", model_names)
    
    if selected_model:
        st.subheader(f"Detailed Evaluation for {selected_model}")
        
        # Confusion matrix
        st.write("Confusion Matrix")
        cm_fig = plot_confusion_matrices({selected_model: st.session_state.evaluation_results[selected_model]})
        st.pyplot(cm_fig)
        
        # ROC Curve
        st.write("ROC Curve")
        roc_fig = plot_roc_curves({selected_model: st.session_state.evaluation_results[selected_model]})
        st.pyplot(roc_fig)
        
        # Feature importance
        st.write("Feature Importance")
        if selected_model in ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']:
            try:
                fi_fig = plot_feature_importance(
                    st.session_state.models[selected_model]['model'],
                    st.session_state.feature_names,
                    model_type=selected_model
                )
                st.pyplot(fi_fig)
            except:
                st.info("Feature importance not available for this model.")
        else:
            st.info("Feature importance not available for this model.")
        
        # Classification report
        st.write("Classification Report")
        report = st.session_state.evaluation_results[selected_model]['classification_report']
        st.text(report)

def render_hyperparameter_tuning_page():
    st.title("Hyperparameter Tuning")
    
    if not st.session_state.models:
        st.warning("Please train your models first in the Model Training section.")
        return
    
    st.write("Perform hyperparameter tuning for your models to improve performance.")
    
    # Select model for tuning
    model_names = list(st.session_state.models.keys())
    selected_model = st.selectbox("Select a model to tune:", model_names)
    
    if selected_model:
        st.subheader(f"Hyperparameter Tuning for {selected_model}")
        
        # Define number of iterations
        n_iter = st.slider("Number of parameter combinations to try:", 5, 50, 10)
        
        # CV folds
        cv_folds = st.slider("Number of cross-validation folds:", 2, 10, 3)
        
        if st.button("Start Hyperparameter Tuning"):
            with st.spinner(f"Tuning {selected_model}... This may take several minutes."):
                try:
                    # Perform hyperparameter tuning
                    best_model, best_params, cv_results = perform_hyperparameter_tuning(
                        selected_model,
                        st.session_state.X_train,
                        st.session_state.y_train,
                        n_iter=n_iter,
                        cv=cv_folds
                    )
                    
                    # Save tuned model
                    st.session_state.tuned_models[selected_model] = {
                        'model': best_model,
                        'params': best_params,
                        'cv_results': cv_results
                    }
                    
                    # Evaluate tuned model
                    tuned_evaluation = evaluate_models(
                        {f"Tuned {selected_model}": {'model': best_model}},
                        st.session_state.X_test,
                        st.session_state.y_test
                    )
                    
                    # Store in session state
                    st.session_state.hp_results[selected_model] = {
                        'tuned_evaluation': tuned_evaluation[f"Tuned {selected_model}"],
                        'original_evaluation': st.session_state.evaluation_results[selected_model]
                    }
                    
                    st.success("Hyperparameter tuning completed successfully!")
                    
                    # Display best parameters
                    st.subheader("Best Parameters")
                    st.json(best_params)
                    
                    # Compare performance before and after tuning
                    st.subheader("Performance Comparison")
                    
                    # Create comparison table
                    original_metrics = st.session_state.evaluation_results[selected_model]
                    tuned_metrics = tuned_evaluation[f"Tuned {selected_model}"]
                    
                    comparison_data = {
                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
                        'Original Model': [
                            original_metrics['accuracy'],
                            original_metrics['precision'],
                            original_metrics['recall'],
                            original_metrics['f1'],
                            original_metrics['roc_auc']
                        ],
                        'Tuned Model': [
                            tuned_metrics['accuracy'],
                            tuned_metrics['precision'],
                            tuned_metrics['recall'],
                            tuned_metrics['f1'],
                            tuned_metrics['roc_auc']
                        ],
                        'Improvement': [
                            tuned_metrics['accuracy'] - original_metrics['accuracy'],
                            tuned_metrics['precision'] - original_metrics['precision'],
                            tuned_metrics['recall'] - original_metrics['recall'],
                            tuned_metrics['f1'] - original_metrics['f1'],
                            tuned_metrics['roc_auc'] - original_metrics['roc_auc']
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df.style.highlight_positive(subset=['Improvement']))
                    
                    # Visualize hyperparameter impact
                    st.subheader("Hyperparameter Tuning Results")
                    hp_fig = plot_hyperparameter_impact(cv_results, selected_model)
                    st.pyplot(hp_fig)
                    
                except Exception as e:
                    st.error(f"Error during hyperparameter tuning: {e}")

def render_comparison_dashboard():
    st.title("Model Comparison Dashboard")
    
    if not st.session_state.evaluation_results:
        st.warning("Please train your models first in the Model Training section.")
        return
    
    st.write("Compare the performance of all trained models in one place.")
    
    # Overall comparison chart
    st.subheader("Performance Metrics Comparison")
    
    # Combine original and tuned models if available
    all_results = st.session_state.evaluation_results.copy()
    for model_name, tuned_info in st.session_state.hp_results.items():
        all_results[f"Tuned {model_name}"] = tuned_info['tuned_evaluation']
    
    metrics_df = plot_model_comparison(all_results)
    st.dataframe(metrics_df)
    
    # Visualize comparison
    st.subheader("Visual Comparison")
    
    # Bar chart comparison
    st.write("Performance Metrics")
    metric_choice = st.selectbox(
        "Select metric to compare:", 
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_values = [results[metric_choice] for results in all_results.values()]
    bars = ax.bar(all_results.keys(), metric_values, color='skyblue')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.title(f'Comparison of {metric_choice.upper()} across models')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(metric_values) * 1.15)  # Add some space above the highest bar
    plt.tight_layout()
    st.pyplot(fig)
    
    # ROC curves comparison
    st.write("ROC Curves Comparison")
    roc_fig = plot_roc_curves(all_results)
    st.pyplot(roc_fig)
    
    # Confusion matrices
    st.write("Confusion Matrices")
    cm_fig = plot_confusion_matrices(all_results)
    st.pyplot(cm_fig)
    
    # Training time comparison
    st.write("Training Time Comparison")
    
    train_times = {}
    for model_name, model_info in st.session_state.models.items():
        train_times[model_name] = model_info['train_time']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(train_times.keys(), train_times.values(), color='lightgreen')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}s', ha='center', va='bottom', rotation=0)
    
    plt.title('Training Time Comparison (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Download model option
    st.subheader("Download Trained Model")
    
    model_to_download = st.selectbox(
        "Select model to download:", 
        list(st.session_state.models.keys()) + [f"Tuned {k}" for k in st.session_state.tuned_models.keys()]
    )
    
    if st.button("Download Model"):
        try:
            # Determine which model to save
            if model_to_download.startswith("Tuned "):
                base_model = model_to_download.replace("Tuned ", "")
                model_obj = st.session_state.tuned_models[base_model]['model']
            else:
                model_obj = st.session_state.models[model_to_download]['model']
            
            # Save model to temporary file
            temp_file = f"{model_to_download.replace(' ', '_').lower()}_model.joblib"
            joblib.dump(model_obj, temp_file)
            
            # Read the file
            with open(temp_file, 'rb') as f:
                model_data = f.read()
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Download button
            st.download_button(
                label="Download Model File",
                data=model_data,
                file_name=temp_file,
                mime="application/octet-stream"
            )
            
        except Exception as e:
            st.error(f"Error downloading model: {e}")

if __name__ == "__main__":
    main()
