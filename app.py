from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from core.scheduling import solve_task1, solve_task2, solve_task3
from core import sampling
from plotly.subplots import make_subplots
import logging
from datetime import datetime
import os
import seaborn as sns
from joblib import load
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import time

# -------------------------------
# 1. Logging Setup
# -------------------------------
def setup_logging():
    """Initialize logging system for audit trail"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f"system_log_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# -------------------------------
# 2. Session State Initialization
# -------------------------------
def init_session_state():
    """Initialize session state variables"""
    if 'app_started' not in st.session_state:
        st.session_state.app_started = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'schedule_results' not in st.session_state:
        st.session_state.schedule_results = {}
    if 'sample_data' not in st.session_state:
        st.session_state.sample_data = None
    if 'selected_records' not in st.session_state:
        st.session_state.selected_records = []
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = None

init_session_state()

# -------------------------------
# 3. Page Configuration
# -------------------------------
st.set_page_config(
    page_title="StaffSync",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
def load_css():
    """Load external CSS file"""
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found. Using default styling.")

load_css()

# -------------------------------
# 4. Helper Functions
# -------------------------------
def log_action(action, details=""):
    """Log user actions to audit trail"""
    logger.info(f"Action: {action} | Details: {details}")

def reset_application():
    """Reset the application to initial state"""
    with st.spinner("Resetting application..."):
        time.sleep(0.5)
        st.session_state.app_started = False
        st.session_state.df = None
        st.session_state.data_uploaded = False
        st.session_state.schedule_results = {}
        st.session_state.sample_data = None
        st.session_state.selected_records = []
        log_action("SYSTEM_RESET", "Application reset to initial state")
        st.rerun()

# =====================================================
# WELCOME/HOME SCREEN
# =====================================================
if not st.session_state.app_started:
    st.markdown("""
        <div class="welcome-card">
            <h1 style="color: white; font-size: 48px; margin-bottom: 20px;">
                StaffSync
            </h1>
            <p style="font-size: 20px; margin-bottom: 0;">
                A&E Optimization & Analytics Platform
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Get Started")
    st.write("Upload your A&E dataset to begin analysis and optimization")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        uploaded_file = st.file_uploader(
            "Upload Dataset (CSV)",
            type=['csv'],
            help="Upload your AED4weeks.csv file or similar dataset"
        )
        
        load_local = st.button("Or use AED4weeks.csv", use_container_width=True)

        if uploaded_file is not None:
            with st.spinner("Loading uploaded file..."):
                try:
                    st.session_state.df = pd.read_csv(uploaded_file)
                    st.session_state.data_uploaded = True
                    time.sleep(0.3)
                    st.success(f"Uploaded file loaded: {uploaded_file.name}")
                    log_action("DATA_UPLOAD", f"File: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error loading uploaded file: {e}")

        if load_local:
            with st.spinner("Loading local dataset..."):
                try:
                    st.session_state.df = pd.read_csv("data/AED4weeks.csv")
                    st.session_state.data_uploaded = True
                    time.sleep(0.3)
                    st.success("Directly loaded AED4weeks.csv from project folder")
                    log_action("LOCAL_DATA_LOAD", "File: AED4weeks.csv")
                except FileNotFoundError:
                    st.error("File 'AED4weeks.csv' not found in project directory.")
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.get('data_uploaded'):
            with st.expander("Preview Data"):
                st.dataframe(st.session_state.df.head(10), use_container_width=True)
            
            st.markdown("---")
            
            if st.button("Launch Application", use_container_width=True, type="primary"):
                with st.spinner("Launching application..."):
                    time.sleep(0.5)
                    st.session_state.app_started = True
                    log_action("APP_START", "User launched application")
                    st.rerun()

else:
    # Application is running
    df = st.session_state.df
    
    # -------------------------------
    # Sidebar Navigation
    # -------------------------------
    with st.sidebar:
        st.title("A&E Operations")
        st.markdown("---")
        
        # Add loading indicator for page changes
        previous_page = st.session_state.get('current_page', None)
        page = st.radio(
            "Navigation Menu",
            ["Dashboard", "Scheduling", "Clinical Analytics", "Sampling", "Breach Prediction", "Data Management"],
            key="navigation_radio"
        )
        
        # Show loading when page changes
        if previous_page and previous_page != page:
            with st.spinner(f"Loading {page}..."):
                time.sleep(0.3)
        
        st.session_state.current_page = page
        
        st.markdown("---")
        
        # System Controls
        st.subheader("System Controls")
        if st.button("Reset Application", use_container_width=True):
            reset_application()
        
        if st.button("Exit Program", use_container_width=True):
            with st.spinner("Exiting..."):
                time.sleep(0.3)
                log_action("APP_EXIT", "User exited application")
                st.session_state.app_started = False
                st.rerun()
        
        st.markdown("---")
        st.caption("University of Manchester")
        st.caption("Operational Research Unit")
        st.caption(f"Records: {len(df):,}")

    # =====================================================
    # PAGE 1: DASHBOARD
    # =====================================================
    if page == "Dashboard":
        st.title("A&E Operations Dashboard")
        log_action("PAGE_VIEW", "Dashboard")
        
        with st.spinner("Loading dashboard metrics..."):
            time.sleep(0.2)
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            breach_rate = (df['Breachornot'] == 'breach').mean() * 100
            
            m1.metric("Total Admissions", f"{len(df):,}")
            m2.metric("Breach Rate", f"{breach_rate:.2f}%")
            m3.metric("Staff Count", "6 Operators")
            m4.metric("Avg Service Time", "14h")

        st.markdown("---")

        st.subheader("Patient Load by HRG Category")
        with st.spinner("Generating chart..."):
            time.sleep(0.2)
            hrg_counts = df["HRG"].value_counts().reset_index()
            hrg_counts.columns = ["HRG", "Count"]
            
            fig = px.bar(hrg_counts, x="HRG", y="Count", 
                         color="Count", color_continuous_scale="Blues",
                         template="plotly_white")
            fig.update_layout(height=600, font_family="Inter", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # PAGE 2: SCHEDULING (Enhanced with Operator View)
    # =====================================================
    elif page == "Scheduling":
        st.title("Operator Schedule Optimization")
        log_action("PAGE_VIEW", "Scheduling")
        
        # Tab change loading indicator
        selected_tab = st.session_state.get('scheduling_tab', 0)
        
        tab1, tab2, tab3 = st.tabs(["Cost Optimization", "Fairness", "Skill Matching"])
        
        with tab1:
            st.subheader("Task 1: Minimize Labor Cost")
            if st.button("Run Cost Optimization", key="run_task1"):
                with st.spinner("Optimizing schedule... This may take a moment."):
                    res, cost = solve_task1()
                    time.sleep(0.5)
                    st.session_state.schedule_results['task1'] = (res, cost)
                    log_action("SCHEDULE_OPTIMIZATION", f"Task 1 - Cost: £{cost:,.2f}")
                    st.success("✓ Optimization complete!")
                    st.rerun()
            
            if 'task1' in st.session_state.schedule_results:
                res, cost = st.session_state.schedule_results['task1']
                st.metric("Total Labor Cost", f"£{cost:,.2f}")
                
                st.markdown("---")
                st.subheader("Complete Schedule")
                st.dataframe(res, use_container_width=True, height=400)
                
                st.markdown("---")
                st.subheader("View Individual Operator Schedule")
                
                if res.index.name or not isinstance(res.index, pd.RangeIndex):
                    operators = sorted(res.index.tolist())
                    
                    selected_operator = st.selectbox(
                        "Select Operator to View Schedule",
                        options=operators,
                        key="task1_operator_select"
                    )
                    
                    if selected_operator:
                        with st.spinner(f"Loading schedule for {selected_operator}..."):
                            time.sleep(0.2)
                            operator_schedule = res.loc[[selected_operator]].copy()
                            
                            st.write(f"**Schedule for {selected_operator}**")
                            st.dataframe(operator_schedule, use_container_width=True, height=150)
                            
                            days = [col for col in res.columns if col != 'Weekly hours']
                            if days:
                                hours_data = operator_schedule[days].values.flatten()
                                fig_bar = go.Figure(data=[
                                    go.Bar(x=days, y=hours_data, marker_color='#0045ac')
                                ])
                                fig_bar.update_layout(
                                    title=f"{selected_operator} - Daily Hours",
                                    xaxis_title="Day",
                                    yaxis_title="Hours",
                                    template="plotly_white",
                                    height=400
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)
                
                elif 'Operator' in res.columns:
                    operators = sorted(res['Operator'].unique())
                    
                    selected_operator = st.selectbox(
                        "Select Operator to View Schedule",
                        options=operators,
                        key="task1_operator_select"
                    )
                    
                    if selected_operator:
                        with st.spinner(f"Loading schedule for {selected_operator}..."):
                            time.sleep(0.2)
                            operator_schedule = res[res['Operator'] == selected_operator].copy()
                            st.write(f"**Schedule for {selected_operator}**")
                            st.dataframe(operator_schedule, use_container_width=True, height=300)
                else:
                    st.info("Unable to identify operator format in results.")
        
        with tab2:
            st.subheader("Task 2: Workload Fairness")
            if st.button("Run Fairness Optimization", key="run_task2"):
                with st.spinner("Optimizing for fairness... This may take a moment."):
                    res, dev = solve_task2()
                    time.sleep(0.5)
                    st.session_state.schedule_results['task2'] = (res, dev)
                    log_action("SCHEDULE_OPTIMIZATION", f"Task 2 - Deviation: {dev:.2f}")
                    st.success("✓ Optimization complete!")
                    st.rerun()
            
            if 'task2' in st.session_state.schedule_results:
                res, dev = st.session_state.schedule_results['task2']
                st.metric("Workload Deviation", f"{dev:.2f}")
                
                st.markdown("---")
                st.subheader("Complete Schedule")
                st.dataframe(res, use_container_width=True, height=400)
                
                st.markdown("---")
                st.subheader("View Individual Operator Schedule")
                
                if res.index.name or not isinstance(res.index, pd.RangeIndex):
                    operators = sorted(res.index.tolist())
                    
                    selected_operator = st.selectbox(
                        "Select Operator to View Schedule",
                        options=operators,
                        key="task2_operator_select"
                    )
                    
                    if selected_operator:
                        with st.spinner(f"Loading schedule for {selected_operator}..."):
                            time.sleep(0.2)
                            operator_schedule = res.loc[[selected_operator]].copy()
                            st.write(f"**Schedule for {selected_operator}**")
                            st.dataframe(operator_schedule, use_container_width=True, height=150)
                            
                            days = [col for col in res.columns if col != 'Weekly hours']
                            if days:
                                hours_data = operator_schedule[days].values.flatten()
                                fig_bar = go.Figure(data=[
                                    go.Bar(x=days, y=hours_data, marker_color='#0045ac')
                                ])
                                fig_bar.update_layout(
                                    title=f"{selected_operator} - Daily Hours",
                                    xaxis_title="Day",
                                    yaxis_title="Hours",
                                    template="plotly_white",
                                    height=400
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)
                
                elif 'Operator' in res.columns:
                    operators = sorted(res['Operator'].unique())
                    selected_operator = st.selectbox(
                        "Select Operator to View Schedule",
                        options=operators,
                        key="task2_operator_select"
                    )
                    
                    if selected_operator:
                        with st.spinner(f"Loading schedule for {selected_operator}..."):
                            time.sleep(0.2)
                            operator_schedule = res[res['Operator'] == selected_operator].copy()
                            st.write(f"**Schedule for {selected_operator}**")
                            st.dataframe(operator_schedule, use_container_width=True, height=300)
                else:
                    st.info("Unable to identify operator format in results.")
        
        with tab3:
            st.subheader("Task 3: Skill-Based Assignment")
            if st.button("Run Skill Optimization", key="run_task3"):
                with st.spinner("Optimizing skill matching... This may take a moment."):
                    res, cost = solve_task3()
                    time.sleep(0.5)
                    st.session_state.schedule_results['task3'] = (res, cost)
                    log_action("SCHEDULE_OPTIMIZATION", f"Task 3 - Cost: £{cost:,.2f}")
                    st.success("✓ Optimization complete!")
                    st.rerun()
            
            if 'task3' in st.session_state.schedule_results:
                res, cost = st.session_state.schedule_results['task3']
                st.metric("Total Cost", f"£{cost:,.2f}")
                
                st.markdown("---")
                st.subheader("Complete Schedule")
                st.dataframe(res, use_container_width=True, height=400)
                
                st.markdown("---")
                st.subheader("View Individual Operator Schedule")
                
                if res.index.name or not isinstance(res.index, pd.RangeIndex):
                    operators = sorted(res.index.tolist())
                    
                    selected_operator = st.selectbox(
                        "Select Operator to View Schedule",
                        options=operators,
                        key="task3_operator_select"
                    )
                    
                    if selected_operator:
                        with st.spinner(f"Loading schedule for {selected_operator}..."):
                            time.sleep(0.2)
                            operator_schedule = res.loc[[selected_operator]].copy()
                            st.write(f"**Schedule for {selected_operator}**")
                            st.dataframe(operator_schedule, use_container_width=True, height=150)
                            
                            days = [col for col in res.columns if col != 'Weekly hours']
                            if days:
                                hours_data = operator_schedule[days].values.flatten()
                                fig_bar = go.Figure(data=[
                                    go.Bar(x=days, y=hours_data, marker_color='#0045ac')
                                ])
                                fig_bar.update_layout(
                                    title=f"{selected_operator} - Daily Hours",
                                    xaxis_title="Day",
                                    yaxis_title="Hours",
                                    template="plotly_white",
                                    height=400
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)
                
                elif 'Operator' in res.columns:
                    operators = sorted(res['Operator'].unique())
                    selected_operator = st.selectbox(
                        "Select Operator to View Schedule",
                        options=operators,
                        key="task3_operator_select"
                    )
                    
                    if selected_operator:
                        with st.spinner(f"Loading schedule for {selected_operator}..."):
                            time.sleep(0.2)
                            operator_schedule = res[res['Operator'] == selected_operator].copy()
                            st.write(f"**Schedule for {selected_operator}**")
                            st.dataframe(operator_schedule, use_container_width=True, height=300)
                else:
                    st.info("Unable to identify operator format in results.")

    # =====================================================
    # PAGE 3: ANALYTICS
    # =====================================================
    elif page == "Clinical Analytics":
        st.title("Clinical Analytics")
        log_action("PAGE_VIEW", "Analytics")

        # =====================================================
        # DESCRIPTIVE ANALYSIS — CLINICAL DATA
        # =====================================================
        st.subheader("Descriptive Analysis")

        numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_vars = df.select_dtypes(include=['object']).columns.tolist()

        c1, c2, c3 = st.columns(3)
        c1.metric("Size", f"{len(df):,}")
        c2.metric("Numeric Variables", len(numeric_vars))
        c3.metric("Categorical Variables", len(categorical_vars))

        # -------------------------------------------------
        # Breach Summary
        # -------------------------------------------------
        if "Breachornot" in df.columns:
            breach_count = (df["Breachornot"] == "breach").sum()
            breach_percent = (breach_count / len(df)) * 100

            st.subheader("Breach Summary")

            b1, b2 = st.columns(2)
            b1.metric("No. of Breaches", f"{breach_count:,}")
            b2.metric("Breach Percentage", f"{breach_percent:.2f}%")

        
        st.subheader("Age Demographics Distribution")
        
        with st.spinner("Generating age distribution chart..."):
            time.sleep(0.3)
            fig_age = go.Figure()
            fig_age.add_trace(go.Histogram(
                x=df["Age"],
                nbinsx=30,
                name='Patient Count',
                marker=dict(
                    color='#457b9d',
                    line=dict(color='white', width=1.5)
                ),
                opacity=0.8
            ))

            fig_age.update_layout(
                template="plotly_white",
                height=700,
                font_family="Inter",
                xaxis_title="Age (Years)",
                yaxis_title="Frequency",
                bargap=0.05
            )
            
            st.plotly_chart(fig_age, use_container_width=True)

        st.markdown("---")

        st.subheader("Analysis: Length of Stay vs. Unit Congestion")
        with st.spinner("Generating scatter plot..."):
            time.sleep(0.3)
            fig_scatter = px.scatter(df, x="noofpatients", y="LoS", 
                                     color="Breachornot",
                                     color_discrete_map={"breach": "#e63946", "no_breach": "#a8dadc"},
                                     template="plotly_white",
                                     size="LoS",
                                     size_max=12)
            fig_scatter.update_layout(height=750, font_family="Inter")
            st.plotly_chart(fig_scatter, use_container_width=True)

    # =====================================================
    # PAGE 4: SAMPLING
    # =====================================================
    elif page == "Sampling":
        st.title("Random Sampling & Statistical Inference")
        log_action("PAGE_VIEW", "Sampling")
        
        st.subheader("Sampling Configuration")

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            max_sample = min(600, len(df))
            if max_sample < 100:
                st.error("Dataset too small for sampling")
                st.stop()
            sample_size = st.slider("Sample Size", 100, max_sample, min(400, max_sample), step=50)
        with c2:
            random_state = st.number_input("Random Seed", value=67)
        with c3:
            st.write("")
            st.write("")
            if st.button("Generate Sample", use_container_width=True):
                with st.spinner("Generating sample..."):
                    time.sleep(0.5)
                    st.session_state.sample_data = df.sample(n=sample_size, random_state=int(random_state))
                    log_action("SAMPLING", f"Size: {sample_size}, Seed: {random_state}")
                    st.rerun()

        if st.session_state.sample_data is not None:
            sample = st.session_state.sample_data
            st.success(f"Sample generated with {len(sample)} records")

            st.markdown("---")

            st.header("Descriptive Analysis")
            
            with st.spinner("Computing statistics..."):
                time.sleep(0.2)
                numeric_vars = sample.select_dtypes(include=[np.number]).columns.tolist()
                categorical_vars = sample.select_dtypes(include=['object']).columns.tolist()

                m1, m2, m3 = st.columns(3)
                m1.metric("Sample Size", len(sample))
                m2.metric("Numeric Variables", len(numeric_vars))
                m3.metric("Categorical Variables", len(categorical_vars))

                # -------------------------------------------------
                # Breach statistics
                # -------------------------------------------------
                st.subheader("Breach Summary (Sample)")

                if "Breachornot" in sample.columns:
                    total_cases = len(sample)
                    breach_count = (sample["Breachornot"] == "breach").sum()
                    breach_percent = (breach_count / total_cases) * 100

                    b1, b2 = st.columns(2)
                    b1.metric("No. of Breaches", f"{breach_count:,}")
                    b2.metric("Breach Percentage", f"{breach_percent:.2f}%")

                st.subheader("Numerical Summary Statistics")
                st.dataframe(sample[numeric_vars].describe().round(2), use_container_width=True, height=400)

                if len(numeric_vars) > 1:
                    st.subheader("Covariance Matrix")
                    st.dataframe(sample[numeric_vars].cov().round(2), use_container_width=True, height=350)

            with st.spinner("Generating correlation heatmap..."):
                time.sleep(0.3)
                corr_matrix = sample[numeric_vars].corr()
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    square=True
                )
                plt.title("Correlation Matrix Heatmap", fontsize=16)
                st.pyplot(plt, use_container_width=True)

            st.markdown("---")
            st.header("Exploratory Visual Analysis")

            st.subheader("Numerical Variable Distributions")
            
            with st.spinner("Generating distribution plots..."):
                time.sleep(0.3)
                available_numeric = ["Age", "LoS", "noofinvestigation", "nooftreatment", "noofpatients"]
                numeric_to_plot = [col for col in available_numeric if col in sample.columns]

                if len(numeric_to_plot) > 0:
                    fig_num = make_subplots(
                        rows=2,
                        cols=3,
                        subplot_titles=numeric_to_plot
                    )

                    for i, col in enumerate(numeric_to_plot):
                        fig_num.add_trace(
                            go.Histogram(
                                x=sample[col],
                                nbinsx=30,
                                marker_color="#457b9d",
                                opacity=0.85
                            ),
                            row=(i // 3) + 1,
                            col=(i % 3) + 1
                        )

                    fig_num.update_layout(
                        template="plotly_white",
                        height=650,
                        showlegend=False,
                        font_family="Inter"
                    )

                    st.plotly_chart(fig_num, use_container_width=True)

            st.markdown("---")

            st.subheader("Categorical Variable Distributions")
            
            with st.spinner("Generating categorical plots..."):
                time.sleep(0.3)
                available_cat = ["DayofWeek", "Breachornot", "HRG"]
                cat_to_plot = [col for col in available_cat if col in sample.columns]

                if len(cat_to_plot) > 0:
                    fig_cat = make_subplots(
                        rows=1,
                        cols=len(cat_to_plot),
                        subplot_titles=cat_to_plot
                    )

                    for i, col in enumerate(cat_to_plot):
                        counts = sample[col].value_counts().reset_index()
                        counts.columns = ["Category", "Count"]

                        fig_cat.add_trace(
                            go.Bar(
                                x=counts["Category"],
                                y=counts["Count"],
                                marker_color="#a8dadc"
                            ),
                            row=1,
                            col=i + 1
                        )

                    fig_cat.update_layout(
                        template="plotly_white",
                        height=450,
                        showlegend=False,
                        font_family="Inter"
                    )

                    st.plotly_chart(fig_cat, use_container_width=True)

            st.markdown("---")
            st.header("Factors Contributing to Breach")
            st.caption("Exploratory comparison between breach and non-breach cases within the sample")

            if "Breachornot" in sample.columns:

                st.subheader("Numerical Factors (mean)")

                numeric_factors = [
                    col for col in ["LoS", "noofpatients", "noofinvestigation", "nooftreatment", "Age"]
                    if col in sample.columns
                ]

                num_summary = (
                    sample
                    .groupby("Breachornot")[numeric_factors]
                    .mean()
                    .T
                    .round(2)
                )

                st.dataframe(num_summary, use_container_width=True)

                st.subheader("Distribution Comparison")

                selected_var = st.selectbox(
                    "Select Variable to Compare",
                    options=numeric_factors
                )

                fig_compare = px.box(
                    sample,
                    x="Breachornot",
                    y=selected_var,
                    color="Breachornot",
                    template="plotly_white",
                    color_discrete_map={
                        "breach": "#e63946",
                        "no_breach": "#457b9d"
                    }
                )

                fig_compare.update_layout(
                    height=500,
                    font_family="Inter",
                    xaxis_title="Outcome",
                    yaxis_title=selected_var
                )

                st.plotly_chart(fig_compare, use_container_width=True)

                st.subheader("Categorical Factors")

                categorical_factors = [
                    col for col in [ "HRG", "Period"]
                    if col in sample.columns
                ]

                selected_cat = st.selectbox(
                    "Select Categorical Variable",
                    options=categorical_factors
                )

                breach_rate_cat = (
                    sample
                    .assign(breach_flag=lambda x: x["Breachornot"] == "breach")
                    .groupby(selected_cat)["breach_flag"]
                    .mean()
                    .reset_index()
                )

                breach_rate_cat["Breach Rate (%)"] = (breach_rate_cat["breach_flag"] * 100).round(2)

                fig_cat_breach = px.bar(
                    breach_rate_cat,
                    x=selected_cat,
                    y="Breach Rate (%)",
                    template="plotly_white",
                    color="Breach Rate (%)",
                    color_continuous_scale="Reds"
                )

                fig_cat_breach.update_layout(
                    height=500,
                    font_family="Inter",
                    yaxis_title="Breach Rate (%)"
                )

                st.plotly_chart(fig_cat_breach, use_container_width=True)

                st.subheader("Key Observations")

                top_numeric = num_summary["Difference (Breach − Non-breach)"].abs().sort_values(ascending=False)

                st.markdown("**Numerical variables most associated with breaches:**")
                for var in top_numeric.head(3).index:
                    st.write(f"• {var}")

                st.markdown(
                    "Higher values of these variables are consistently observed in breach cases, "
                    "suggesting increased workload and patient complexity as primary drivers."
                )

            st.markdown("---")

            if "HRG" in sample.columns:
                st.subheader("HRG Category Distribution")
                
                with st.spinner("Generating HRG distribution..."):
                    time.sleep(0.2)
                    hrg_counts = sample["HRG"].value_counts().reset_index()
                    hrg_counts.columns = ["HRG", "Patient Count"]

                    fig_hrg = px.bar(
                        hrg_counts,
                        x="HRG",
                        y="Patient Count",
                        template="plotly_white",
                        color="Patient Count",
                        color_continuous_scale="Blues"
                    )

                    fig_hrg.update_layout(
                        height=500,
                        font_family="Inter",
                        showlegend=False,
                        yaxis_title="Number of Patients"
                    )

                    st.plotly_chart(fig_hrg, use_container_width=True)

            st.markdown("---")

            if "Period" in sample.columns:
                st.subheader("Arrivals Distribution by Time of Day")
                
                with st.spinner("Generating time period analysis..."):
                    time.sleep(0.2)
                    sample["HourPeriod"] = pd.cut(
                        sample["Period"],
                        bins=[0, 6, 12, 18, 24],
                        labels=["Night", "Morning", "Afternoon", "Evening"],
                        right=False
                    )

                    period_counts = sample["HourPeriod"].value_counts().reset_index()
                    period_counts.columns = ["Hour Period", "Arrivals"]

                    fig_pie = px.pie(
                        period_counts,
                        names="Hour Period",
                        values="Arrivals",
                        hole=0.35,
                        template="plotly_white"
                    )

                    fig_pie.update_layout(
                        height=450,
                        font_family="Inter",
                        title="Arrivals by Hour Period"
                    )

                    st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")

            if "Age" in sample.columns and "LoS" in sample.columns:
                st.subheader("Length of Stay vs Age")

                with st.spinner("Generating scatter analysis..."):
                    time.sleep(0.2)
                    fig_los_age = px.scatter(
                        sample,
                        x="Age",
                        y="LoS",
                        color="Breachornot" if "Breachornot" in sample.columns else None,
                        template="plotly_white",
                        opacity=0.7,
                        color_discrete_map={
                            "breach": "#e63946",
                            "non-breach": "#457b9d"
                        } if "Breachornot" in sample.columns else None
                    )

                    fig_los_age.update_layout(
                        height=600,
                        font_family="Inter",
                        xaxis_title="Age (Years)",
                        yaxis_title="Length of Stay"
                    )

                    st.plotly_chart(fig_los_age, use_container_width=True)
        else:
            st.info("Click 'Generate Sample' to begin analysis")

    # =====================================================
    # PAGE 5: BREACH PREDICTION
    # =====================================================
    elif page == "Breach Prediction":
        st.title("Breach Prediction System")
        log_action("PAGE_VIEW", "Breach Prediction")
        
        # Load model
        if not st.session_state.model_loaded:
            try:
                st.session_state.model = load("xgb_model.pkl")
                st.session_state.model_loaded = True
                st.success("Model loaded successfully")
                log_action("MODEL_LOAD", "XGBoost model loaded")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.info("Please ensure 'xgb_model.pkl' is in the working directory")
                log_action("MODEL_LOAD_ERROR", f"Error: {str(e)}")
        
        if st.session_state.model_loaded:
            st.markdown("---")
            st.subheader("Enter Patient Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Numerical Features**")
                age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
                period = st.number_input("Period (Hour of Day)", min_value=0, max_value=23, value=12, step=1)
                noofinvestigation = st.number_input("Number of Investigations", min_value=0, max_value=20, value=2, step=1)
            
            with col2:
                st.markdown("**Numerical Features (cont.)**")
                nooftreatment = st.number_input("Number of Treatments", min_value=0, max_value=20, value=1, step=1)
                noofpatients = st.number_input("Number of Patients in Unit", min_value=0, max_value=100, value=25, step=1)
            
            with col3:
                st.markdown("**Categorical Features**")
                dayofweek = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
                hrg = st.selectbox("HRG Category", ["Minor", "Major", "Resuscitation"])
                day = st.number_input("Day of Month", min_value=1, max_value=31, value=15, step=1)

            # Predict button
            col_center = st.columns([1, 1, 1])[1]
            with col_center:
                if st.button("Predict Breach", use_container_width=True, type="primary"):
                    try:
                        # Create input dataframe with ONLY the features used in training
                        # Match exactly what the model was trained on
                        input_data = pd.DataFrame({
                            "Age": [age],
                            "Period": [period],
                            "noofinvestigation": [noofinvestigation],
                            "nooftreatment": [nooftreatment],
                            "noofpatients": [noofpatients],
                            "DayofWeek": [dayofweek],
                            "HRG": [hrg],
                            "Day": [day]
                        })
                        
                        # The model is a Pipeline with preprocessing built-in
                        # Just pass the dataframe directly
                        prediction_proba = st.session_state.model.predict_proba(input_data)[0]
                        prediction = st.session_state.model.predict(input_data)[0]
                        
                        log_action("BREACH_PREDICTION", f"Prediction: {prediction}, Probability: {prediction_proba}")

                        st.subheader("Prediction Results")

                        st.markdown(f"""
                        <div class="prediction-container">
                            <div class="prediction-card danger">
                                <div class="prediction-title">Chance of Breach</div>
                                <div class="prediction-value">
                                    {prediction_proba[1]*100:.1f}%
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Probability visualization
                        st.markdown("---")
                        fig_prob = go.Figure(data=[
                            go.Bar(
                                x=['No Breach', 'Breach'],
                                y=[prediction_proba[0], prediction_proba[1]],
                                marker_color=['#a8dadc', '#e63946'],
                                text=[f"{prediction_proba[0]*100:.1f}%", f"{prediction_proba[1]*100:.1f}%"],
                                textposition='auto'
                            )
                        ])
                        fig_prob.update_layout(
                            title="Prediction Probabilities",
                            yaxis_title="Probability",
                            template="plotly_white",
                            height=400,
                            font_family="Inter"
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        log_action("PREDICTION_ERROR", f"Error: {str(e)}")
                        st.info("Please ensure all input features match the model's training data")

    # =====================================================
    # PAGE 6: DATA MANAGEMENT (Enhanced)
    # =====================================================
    elif page == "Data Management":
        st.title("Advanced Data Management")
        log_action("PAGE_VIEW", "Data Management")
        
        tab1, tab2, tab3 = st.tabs(["Search & Filter", "Range Operations", "Audit Log"])
        
        with tab1:
            st.subheader("Search and Filter Records")
            search = st.text_input("Filter by Patient ID")
            filtered = df if not search else df[df["ID"].str.contains(search, case=False, na=False)]
            
            st.write(f"Showing {len(filtered)} of {len(df)} records")
            st.dataframe(filtered, use_container_width=True, height=500)
        
        with tab2:
            st.subheader("Range-Based Data Operations")
            
            # Operation Panel at the top
            st.markdown("#### Step 1: Define Range Filter")
            
            # Select variable
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_var = st.selectbox("Select Variable", numeric_cols)
            
            if selected_var:
                col1, col2 = st.columns(2)
                with col1:
                    min_val = st.number_input(
                        "Minimum Value",
                        value=float(df[selected_var].min()),
                        step=1.0
                    )
                with col2:
                    max_val = st.number_input(
                        "Maximum Value",
                        value=float(df[selected_var].max()),
                        step=1.0
                    )
                
                # Filter data in range
                range_filtered = df[
                    (df[selected_var] >= min_val) & 
                    (df[selected_var] <= max_val)
                ]
                
                st.info(f"**Found {len(range_filtered)} records where {selected_var} is between {min_val} and {max_val}**")
                
                st.markdown("---")

                st.markdown("#### Preview: Records in Range")
                st.dataframe(range_filtered, use_container_width=True, height=300)

                st.markdown("---")

                st.markdown("#### Step 2: Select Records to Modify")
                st.caption("👇 Scroll down to view and select records from the filtered results below")
                
                # Show filtered records with checkboxes
                if len(range_filtered) > 0:
                    # Select all option
                    select_all = st.checkbox("Select All Records", value=False)
                    
                    if select_all:
                        st.session_state.selected_records = range_filtered.index.tolist()
                    else:
                        st.session_state.selected_records = []
                    
                    # Individual selection
                    st.write("**Or select individual records:**")
                    selected_indices = st.multiselect(
                        "Choose records by ID",
                        options=range_filtered.index.tolist(),
                        format_func=lambda x: f"ID: {df.loc[x, 'ID']} | {selected_var}: {df.loc[x, selected_var]}",
                        default=st.session_state.selected_records if not select_all else range_filtered.index.tolist()
                    )
                    
                    if not select_all:
                        st.session_state.selected_records = selected_indices
                    
                    st.write(f"**Selected: {len(st.session_state.selected_records)} records**")
                    
                    st.markdown("---")
                    st.markdown("#### Step 3: Perform Operations")
                    
                    op_col1, op_col2 = st.columns(2)
                    
                    with op_col1:
                        st.markdown("**Delete Records**")
                        if st.button("Delete Selected Records", use_container_width=True, type="primary"):
                            if len(st.session_state.selected_records) > 0:
                                num_deleted = len(st.session_state.selected_records)
                                st.session_state.df = df.drop(st.session_state.selected_records)
                                log_action(
                                    "DATA_DELETE",
                                    f"Deleted {num_deleted} records where {selected_var} ∈ [{min_val}, {max_val}]"
                                )
                                st.success(f"Deleted {num_deleted} records")
                                st.session_state.selected_records = []
                                st.rerun()
                            else:
                                st.warning("No records selected")
                    
                    with op_col2:
                        st.markdown("**Modify Values**")
                        new_value = st.number_input("New Value for Selected Variable", step=1.0, key="new_val")
                        if st.button("Update Selected Records", use_container_width=True):
                            if len(st.session_state.selected_records) > 0:
                                num_updated = len(st.session_state.selected_records)
                                old_values = st.session_state.df.loc[st.session_state.selected_records, selected_var].tolist()
                                st.session_state.df.loc[st.session_state.selected_records, selected_var] = new_value
                                log_action(
                                    "DATA_MODIFY",
                                    f"Updated {num_updated} records: {selected_var} changed from {old_values[:3]}{'...' if len(old_values) > 3 else ''} to {new_value}"
                                )
                                st.success(f"Updated {num_updated} records")
                                st.session_state.selected_records = []
                                st.rerun()
                            else:
                                st.warning("No records selected")
                    
                    st.markdown("---")
                    
                    if st.button("Clear Selection", use_container_width=False):
                        st.session_state.selected_records = []
                        st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
        
        with tab3:
            st.subheader("System Audit Log")
            
            log_file = f"logs/system_log_{datetime.now().strftime('%Y%m%d')}.log"
            
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    log_content = f.readlines()

                st.write(f"**Log File:** {log_file}")
                st.write(f"**Total Entries:** {len(log_content)}")
                
                # Display recent logs
                st.text_area(
                    "Recent Activity",
                    value="".join(log_content[-50:]),
                    height=500
                )
                
                # Download log
                if st.download_button(
                    "Download Full Log",
                    data="".join(log_content),
                    file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                ):
                    log_action("LOG_DOWNLOAD", "User downloaded audit log")
            else:
                st.info("No log file found for today")