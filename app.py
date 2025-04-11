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
    
    # Default to sample dataset for a better demo experience
    data_option = st.radio(
        "Choose data source:",
        ("Use sample Telco Customer Churn dataset", "Upload CSV file")
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
            # Display a spinner while loading the Telco dataset
            with st.spinner("Loading Telco Customer Churn dataset..."):
                url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
                data = pd.read_csv(url)
                st.success("Telco Customer Churn dataset loaded successfully!")
                
                # Add dataset information for context
                with st.expander("About Telco Customer Churn Dataset", expanded=False):
                    st.markdown("""
                    ### Dataset Information
                    
                    The **Telco Customer Churn** dataset contains information about a fictional telco company that provides home phone and internet services to customers. The data indicates which customers have left, stayed, or signed up for their service.
                    
                    #### Key Features:
                    - **Demographics**: Customer gender, age range, and if they have partners and dependents
                    - **Account Information**: Tenure, contract type, payment method, paperless billing
                    - **Services**: Phone, multiple lines, internet service, online security, online backup, device protection, tech support, and streaming TV and movies
                    
                    #### Target Variable:
                    - **Churn**: Whether the customer left within the last month
                    
                    This is one of the most widely used datasets for binary classification machine learning problems.
                    """)
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
            return
    
    if data is not None:
        st.session_state.data = data
        
        # Show dataset in tabs for better organization
        tab1, tab2, tab3 = st.tabs(["Data Preview", "Data Information", "Statistical Summary"])
        
        with tab1:
            st.dataframe(data.head(10), use_container_width=True)
            
            # Show basic stats about the dataset
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{data.shape[0]:,}")
            with col2:
                st.metric("Columns", data.shape[1])
            with col3:
                st.metric("Missing Values", f"{data.isna().sum().sum():,}")
        
        with tab2:
            buffer = StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
            
            # Show column types
            col_types = data.dtypes.value_counts().reset_index()
            col_types.columns = ['Data Type', 'Count']
            st.subheader("Column Data Types")
            st.dataframe(col_types)
            
        with tab3:
            st.dataframe(data.describe(include='all'), use_container_width=True)
        
        st.subheader("Data Preparation")
        
        # Pre-select appropriate values for Telco dataset if it's being used
        is_telco = False
        if data_option == "Use sample Telco Customer Churn dataset":
            is_telco = True
            default_target = "Churn"
            default_drop = ["customerID"]
            
            # Use knowledge of Telco dataset to pre-select columns
            default_categorical = [
                'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                'PaperlessBilling', 'PaymentMethod'
            ]
            
            default_numerical = [
                'tenure', 'MonthlyCharges', 'TotalCharges'
            ]
            
            # Information box about column types
            st.info("""
            #### Column Type Information:
            
            - **Target Column**: The main variable we're trying to predict (Churn - whether a customer left the company)
            - **Categorical Columns**: Variables with distinct categories (like gender, contract type)
            - **Numerical Columns**: Quantitative measurements (like tenure, monthly charges)
            - **Columns to Drop**: Variables not useful for analysis (like customerID)
            """)
        else:
            default_target = ""
            default_drop = [col for col in data.columns if col.lower() in ['id', 'customerid', 'customer_id']]
            default_categorical = []
            default_numerical = []
        
        # Select target column with appropriate default
        if is_telco:
            target_column = st.selectbox(
                "Select target column:", 
                data.columns.tolist(), 
                index=data.columns.get_loc(default_target) if default_target in data.columns else 0
            )
        else:
            target_column = st.selectbox("Select target column:", data.columns.tolist())
            
        st.session_state.target_column = target_column
        
        # Handle columns to drop with appropriate defaults
        columns_to_drop = st.multiselect(
            "Select columns to drop (e.g., ID columns):", 
            data.columns.tolist(), 
            default=[col for col in default_drop if col in data.columns]
        )
        
        # Handle categorical columns with appropriate defaults
        if is_telco:
            categorical_candidates = [col for col in data.columns if col not in columns_to_drop and col != target_column and data[col].dtype == 'object']
            categorical_columns = st.multiselect(
                "Select categorical columns:",
                categorical_candidates,
                default=[col for col in default_categorical if col in categorical_candidates]
            )
        else:
            categorical_columns = st.multiselect(
                "Select categorical columns:",
                [col for col in data.columns if col not in columns_to_drop and col != target_column and data[col].dtype == 'object']
            )
        st.session_state.categorical_columns = categorical_columns
        
        # Handle numerical columns with appropriate defaults
        if is_telco:
            numerical_candidates = [col for col in data.columns if col not in columns_to_drop and col != target_column and data[col].dtype != 'object']
            numerical_columns = st.multiselect(
                "Select numerical columns:",
                numerical_candidates,
                default=[col for col in default_numerical if col in numerical_candidates]
            )
        else:
            numerical_columns = st.multiselect(
                "Select numerical columns:",
                [col for col in data.columns if col not in columns_to_drop and col != target_column and data[col].dtype != 'object']
            )
        st.session_state.numerical_columns = numerical_columns
        
        # Show feature selection summary
        if categorical_columns or numerical_columns:
            st.success(f"Selected {len(categorical_columns)} categorical features and {len(numerical_columns)} numerical features for modeling")
            
            # Visual indicator of selected features
            feature_count = len(categorical_columns) + len(numerical_columns)
            total_count = len(data.columns) - len(columns_to_drop) - 1  # Excluding target and dropped columns
            st.progress(feature_count / total_count if total_count > 0 else 0)
        
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
    st.title("🏆 Model Comparison Dashboard")
    
    if not st.session_state.evaluation_results:
        st.warning("Please train your models first in the Model Training section.")
        return
    
    # Add a descriptive introduction with explanation of this page
    st.markdown("""
    <div class="info-box">
    <h4 style="margin-top:0">Comprehensive Model Analysis</h4>
    <p>This dashboard provides side-by-side comparison of all trained models, allowing you to quickly identify the best performing models
    across different evaluation metrics. Compare standard models against tuned variations to see the impact of hyperparameter optimization.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "📈 Visualizations", "⚙️ Implementation Details"])
    
    # Combine original and tuned models if available
    all_results = st.session_state.evaluation_results.copy()
    for model_name, tuned_info in st.session_state.hp_results.items():
        all_results[f"Tuned {model_name}"] = tuned_info['tuned_evaluation']
    
    with tab1:
        st.subheader("Performance Metrics Comparison")
        
        # Create a more visually appealing styled metrics dataframe
        metrics_df = plot_model_comparison(all_results)
        
        # Determine best model for each metric
        best_model = {}
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
            if metric in metrics_df.columns:
                best_idx = metrics_df[metric].idxmax()
                best_model[metric] = (best_idx, metrics_df.loc[best_idx, metric])
        
        # Display metrics dataframe with styling
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='#DCFCE7'), use_container_width=True)
        
        # Show summary cards for quick reference
        if best_model:
            st.markdown("### 🏆 Best Model by Metric")
            
            # Create summary metrics cards
            cols = st.columns(len(best_model))
            for i, (metric, (model, value)) in enumerate(best_model.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div style="padding: 1rem; background: linear-gradient(90deg, rgba(53, 99, 233, 0.1), rgba(53, 99, 233, 0.05)); 
                                border-radius: 0.5rem; border-left: 4px solid #3563E9; text-align: center;">
                        <div style="font-size: 0.85rem; color: #6B7280; margin-bottom: 0.5rem;">{metric}</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #1E3A8A; margin-bottom: 0.5rem;">{value:.4f}</div>
                        <div style="font-size: 0.9rem; color: #374151; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{model}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show interactive metric exploration
        st.markdown("### 📊 Metric Details")
        selected_metric = st.selectbox(
            "Select a metric to explore in depth:", 
            ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'], 
            index=0
        )
        
        # Show percent difference from the best model
        if selected_metric in metrics_df.columns:
            best_value = metrics_df[selected_metric].max()
            st.markdown(f"**Best {selected_metric}:** {best_value:.4f} (Model: {metrics_df[selected_metric].idxmax()})")
            
            # Calculate percent differences for comparison
            comp_df = pd.DataFrame({
                'Model': metrics_df.index,
                'Value': metrics_df[selected_metric],
                'Difference from Best (%)': 100 * (metrics_df[selected_metric] - best_value) / best_value
            }).sort_values('Value', ascending=False)
            
            st.dataframe(comp_df.style.format({
                'Value': '{:.4f}',
                'Difference from Best (%)': '{:.2f}%'
            }).background_gradient(subset=['Value'], cmap='Blues'), use_container_width=True)
    
    with tab2:
        # Bar chart comparison with improved styling
        st.subheader("Performance Visualization")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            metric_choice = st.selectbox(
                "Select metric to compare:", 
                ["accuracy", "precision", "recall", "f1", "roc_auc"],
                format_func=lambda x: {
                    "accuracy": "Accuracy", 
                    "precision": "Precision", 
                    "recall": "Recall", 
                    "f1": "F1 Score", 
                    "roc_auc": "ROC AUC"
                }[x]
            )
        
        with col2:
            # Add some visual enhancements options
            show_value_labels = st.checkbox("Show value labels", value=True)
            
        # Apply custom styling to make more visually appealing
        from utils import set_matplotlib_style
        colors = set_matplotlib_style()
            
        # Create enhanced bar chart
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        fig.patch.set_facecolor('white')
        
        # Get metric values and sort from highest to lowest
        model_metrics = [(model, results[metric_choice]) for model, results in all_results.items()]
        model_metrics.sort(key=lambda x: x[1], reverse=True)
        
        models, metric_values = zip(*model_metrics)
        
        # Generate gradient color palette for models
        color_mapping = {}
        for i, model in enumerate(models):
            if "Tuned" in model:
                # Use a specific color for tuned models to differentiate them
                color_mapping[model] = colors[1]  # Use second color for tuned models
            else:
                color_mapping[model] = colors[0]  # Use first color for base models
        
        model_colors = [color_mapping[model] for model in models]
            
        # Create the bar chart with enhanced styling
        bars = ax.bar(
            models, metric_values, 
            color=model_colors,
            edgecolor='white',
            linewidth=1.5,
            alpha=0.8
        )
        
        # Highlight best model with additional marker
        best_idx = metric_values.index(max(metric_values))
        bars[best_idx].set_color(colors[2])  # Use third color for best model
        bars[best_idx].set_alpha(1.0)
        
        # Add a subtle grid
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels if chosen
        if show_value_labels:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + 0.005,
                    f'{height:.4f}', 
                    ha='center', 
                    va='bottom', 
                    fontsize=10,
                    fontweight='bold',
                    color='#444'
                )
        
        # Enhanced styling
        ax.set_title(f'Comparison of {metric_choice.upper()} across models', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Models', fontsize=12, fontweight='semibold', labelpad=10)
        ax.set_ylabel(f'{metric_choice.title()} Score', fontsize=12, fontweight='semibold', labelpad=10)
        
        # Style x-axis ticks
        plt.xticks(rotation=45, ha='right')
        
        # Add some space above the highest bar
        plt.ylim(0, max(metric_values) * 1.15)  
        
        # Add explanatory text
        if "Tuned" in "".join(models):
            ax.text(
                0.5, 0.02, 
                "Note: 'Tuned' models have undergone hyperparameter optimization", 
                transform=fig.transFigure,
                ha='center',
                fontsize=10,
                fontstyle='italic',
                color='#666'
            )
        
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
