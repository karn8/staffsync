# staffsync
StaffSync
A&E Operations Optimisation and Analytics Platform

StaffSync is a Streamlit-based analytical and decision-support system designed for Accident and Emergency (A&E) departments. The platform integrates operations research, statistical analysis, and machine learning to support operational planning, performance monitoring, and breach risk prediction.

The application is intended for academic, analytical, and operational use, with an emphasis on transparency, auditability, and reproducibility.

Project Objectives

The primary objectives of StaffSync are to:

Analyse operational performance in A&E departments

Optimise staff scheduling under multiple constraints and objectives

Provide statistical insight through controlled sampling and inference

Predict four-hour breach risk using a trained machine learning model

Maintain a complete audit trail of system interactions and data changes

System Architecture

The system is implemented as a modular Streamlit application with a clear separation between user interface logic, optimisation models, data handling, and auditing.

├── app.py                     # Main Streamlit application
├── core/
│   ├── __init__.py
│   ├── audit_logger.py        # Centralised audit logging
│   ├── data_manager.py        # Data loading and validation
│   ├── data.py                # Data utilities
│   ├── sampling.py            # Sampling and inference logic
│   ├── scheduling.py          # Optimisation models (Tasks 1–3)
│   └── utils.py               # Shared helper functions
├── data/
│   ├── AED4weeks.csv          # Sample A&E dataset
│   └── operators.csv          # Operator availability and skills
├── logs/
│   └── system_log_YYYYMMDD.log
├── model.py                   # Model training pipeline
├── xgb_model.pkl              # Trained XGBoost model
└── README.md

Application Workflow

The user launches the application and uploads an A&E dataset or loads the provided sample dataset.

Data is validated and stored in session state.

The user navigates between functional modules via the sidebar.

All actions are logged to a timestamped audit file.

Optimisation, analytics, and predictions are performed interactively.

Functional Modules
Dashboard

Overview of admissions, breach rates, staffing levels, and service times

Visual analysis of patient load by HRG category

Scheduling Optimisation

Three optimisation models are provided:

Cost minimisation

Workload fairness

Skill-based staff assignment

Each model supports:

Full schedule display

Operator-level inspection

Visual daily workload plots

Optimisation models are implemented using linear programming techniques.

Clinical Analytics

Age distribution analysis

Length of stay versus congestion analysis

Interactive scatter plots and histograms

Sampling and Statistical Inference

Random sampling with reproducible seeds

Descriptive statistics

Covariance and correlation analysis

Distribution visualisation for numerical and categorical variables

Time-of-day arrival analysis

Breach Prediction

Machine learning–based four-hour breach prediction

Trained XGBoost model with preprocessing pipeline

Probability-based outputs with visual interpretation

Manual feature input for scenario testing

Data Management

Search and filter records

Range-based selection and modification

Controlled deletion and updates

Full audit logging of all data operations

Audit Logging

All system actions are logged, including:

Data uploads and loads

Optimisation runs

Predictions

Data modifications and deletions

System resets and exits

Logs are stored in the logs/ directory and can be viewed or downloaded directly from the application.

Technologies Used

Python

Streamlit

Pandas, NumPy

Plotly, Matplotlib, Seaborn

Scikit-learn

XGBoost

PuLP (Linear Programming)

Joblib

Logging (Python standard library)

Running the Application
Prerequisites

Python 3.9 or higher

All required Python packages installed

Installation
pip install -r requirements.txt

Launch
streamlit run app.py

Data

The repository includes a sample dataset (AED4weeks.csv) representing four weeks of A&E activity. Users may also upload their own datasets, provided the schema matches the expected format.

It is not intended for direct clinical deployment without appropriate validation and governance.

Author and Affiliation

Developed as part of academic work in BMAN73701  
Python Programming for Business Intelligence and Analytics 
(2025-2026)

-Karan Gupta
github.com/karn8

-Piyush Pethkar
github.com/piyushpethkar11
