Absolutely! Here's a **detailed and beginner-friendly version** of your README file that assumes **no prior coding knowledge**. It explains each step clearly, with friendly guidance and optional instructions for both Windows and macOS/Linux users.

---

# 🚀 Machine Learning Model Comparison Dashboard

An interactive and easy-to-use web application for **comparing and visualizing classification models** with **automated hyperparameter tuning** — designed for students, researchers, and data enthusiasts.

---

## 🌟 Key Features

- 📂 **Data Preparation**  
  Upload your own CSV file or use the built-in Telco Customer Churn dataset to get started.

- 🧠 **Model Training**  
  Train five popular classification models with a single click.

- 📊 **Model Evaluation**  
  Get detailed performance reports: accuracy, precision, recall, F1 score, AUC, confusion matrix, and more.

- 🛠️ **Hyperparameter Tuning**  
  Automatically find the best model parameters for better performance.

- 📈 **Comparison Dashboard**  
  Compare trained models using charts and metrics on an interactive dashboard.

---

## 🤖 Models You Can Compare

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost

---

## 📏 Evaluation Metrics Available

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC Curve & AUC  
- Confusion Matrix  
- Feature Importance Charts

---

## 🧰 How to Run the App (No Coding Experience Needed!)

This guide will help you run the app on your own computer — even if you've never coded before!

---

### ✅ Step 1: Install Python (if not already installed)

1. Visit the [official Python website](https://www.python.org/downloads/)  
2. Download the latest version for your operating system (Windows or macOS)
3. During installation, make sure you check the box that says:  
   **"Add Python to PATH"**

---

### ✅ Step 2: Clone the Project (Get the App Code)

> 💡 You can either use **Git** or just download the ZIP file.

#### Option A: Using Git (Recommended)
1. Install Git from [git-scm.com](https://git-scm.com/)
2. Open a terminal (Command Prompt or PowerShell on Windows, Terminal on macOS/Linux)
3. Run the following:
```bash
git clone https://github.com/MdIrfan325/ModelComparission.git
cd ModelComparer
```

#### Option B: Without Git (Manual Download)
1. Visit: https://github.com/your-username/ModelComparer
2. Click the green **Code** button → then **Download ZIP**
3. Extract the ZIP file
4. Open the extracted folder in your terminal or command prompt

---

### ✅ Step 3: Install Required Packages

In the terminal, run:
```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, you can run this instead:
```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

---

### ✅ Step 4: Run the App

In the same terminal, type:
```bash
streamlit run app.py
```

Then, open the link it gives you (usually `http://localhost:8501`) in your **web browser**.

> 🔗 You can also visit your deployed version of the app here:  
[Open Web App](https://opulent-space-waffle-69gj546x5rp43xxwv-5000.app.github.dev/)

---

## 🎯 How to Use the App (Interface Guide)

1. **Home**  
   Read a summary of what the app does.

2. **Data Preparation**  
   Upload a CSV file or use the sample dataset provided.

3. **Model Training**  
   Select models to train and click the train button.

4. **Model Evaluation**  
   View the confusion matrix, classification report, and ROC curves.

5. **Hyperparameter Tuning**  
   Tune model parameters for better accuracy automatically.

6. **Comparison Dashboard**  
   View all model performances side-by-side with graphs and metrics.

7. **Export Models**  
   Download any trained or tuned model in `.joblib` format.
   when you click on download in the sidebar a download button is generated in the bottom of the comparision dashboard when clicked your model is downloaded.
   ![Screenshot 2025-04-12 173926](https://github.com/user-attachments/assets/2dfd7f16-2605-4461-b947-8d9b3491efd2)


---

### video presentation
Google Drive Link: https://drive.google.com/file/d/1EBipxXT-EJRMB0O-3Avtx6d6TDfBwOA2/view?usp=sharing


---
---

### Sample Data Used:
https://drive.google.com/file/d/1ORi3PG4HfGWSS6pVMC533JS6CG-_Npps/view?usp=drive_link

---
---

## 👨‍💻 Authors

**Mohammed Irfan**  
📧 mi3253050@gmail.com

**Enumula Umamaheshwari**  
📧 umaenumula04@gmail.com

---

## 💡 Tips

- The app works best with clean, well-formatted CSV files.
- You can run the app locally or host it using platforms like Streamlit Cloud.
- No coding required once the app is running!

