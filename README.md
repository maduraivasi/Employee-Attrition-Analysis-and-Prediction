Employee Attrition Analysis and Prediction
📌 Project Overview
Employee turnover is a costly challenge that impacts productivity and organizational stability. This project leverages Data Science and Machine Learning to analyze 1,470+ employee records, identify the root causes of attrition, and provide a predictive tool for HR departments.

By using historical data—including demographics, job roles, and satisfaction levels—this project builds a robust classification model to predict the likelihood of an employee leaving the company.

🚀 Features
Exploratory Data Analysis (EDA): Visual insights into how factors like salary, age, and overtime affect attrition.

Predictive Modeling: Machine Learning models (Random Forest/Logistic Regression) to classify at-risk employees.

Interactive Dashboard: A Streamlit web application that allows HR managers to input employee data and get real-time predictions.

Actionable Insights: Identification of key drivers (e.g., Job Satisfaction, Monthly Income) to help in retention planning.

🛠️ Tech Stack
Languages: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Deployment: Streamlit

Model Storage: Pickle/Joblib

📂 Project Structure
Plaintext
├── data/
│   ├── employee_attrition.csv       # Original Dataset
│   └── cleaned_data.csv            # Preprocessed Data
├── models/
│   └── attrition_model.pkl         # Trained Machine Learning Model
├── notebooks/
│   └── EDA_and_Modeling.ipynb      # Jupyter Notebook with full analysis
├── app.py                          # Streamlit Application Script
├── requirements.txt                # List of dependencies
└── README.md                       # Project Documentation
📊 Key Findings from Data
Overtime: Employees working overtime show a significantly higher rate of attrition.

Monthly Income: Lower income brackets are more prone to leaving.

Age: Younger employees (under 30) tend to have higher turnover rates.

Job Satisfaction: Ratings of '1' (Low) are a primary indicator of potential exit.
