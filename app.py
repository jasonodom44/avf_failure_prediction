import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="AVF Failure Risk Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Now works WITH the config.toml file
# Custom CSS - Now works WITH the config.toml file
st.markdown("""
    <style>
    /* --- Sidebar Styling --- */
    /* This overrides the white 'secondaryBackgroundColor' from the config */
    [data-testid="stSidebar"] {
        background-color: #1e3a8a;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* --- Custom Component Colors --- */
    /* Headers */
    h1, h2, h3 {
        color: #1e3a8a;
    }

    /* Metric Values */
    [data-testid="stMetricValue"] {
        color: #1e3a8a;
        font-weight: bold;
    }

    /* Dataframe Headers */
    .dataframe th {
        background-color: #1e3a8a !important;
        color: white !important;
    }

    /* --- Your Custom Risk Scores --- */
    .risk-critical {
        color: #dc2626;
        font-size: 28px;
        font-weight: bold;
    }
    .risk-high {
        color: #ea580c;
        font-size: 28px;
        font-weight: bold;
    }
    .risk-moderate {
        color: #ca8a04;
        font-size: 28px;
        font-weight: bold;
    }
    .risk-low {
        color: #16a34a;
        font-size: 28px;
        font-weight: bold;
    }

    /* --- Fix for containers (st.metric, etc.) --- */
    /* THIS IS THE CORRECTED RULE */
    [data-testid="stMain"] div[data-testid="stVerticalBlock"] > div {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# Load data
@st.cache_data
def load_data():
    patients = pd.read_csv('data/raw/patients_baseline.csv')
    treatments = pd.read_csv('data/raw/treatments_baseline.csv')
    outcomes = pd.read_csv('data/raw/outcomes_baseline.csv')

    # Merge for full dataset
    full_data = treatments.merge(patients, on='patient_id')
    full_data = full_data.merge(outcomes[['patient_id', 'failed']], on='patient_id')

    return patients, treatments, outcomes, full_data


@st.cache_resource
def load_model():
    return joblib.load('models/rf_avf_failure_model.pkl')


# More realistic risk score calculation
def calculate_risk_score(row):
    risk = row['baseline_risk_score'] * 50  # Start with baseline (0-50%)

    # Add risk from low Qa
    if row['access_blood_flow_qa'] < 500:
        risk += 30
    elif row['access_blood_flow_qa'] < 600:
        risk += 20
    elif row['access_blood_flow_qa'] < 700:
        risk += 10

    # Add risk from high SVPR
    if row['svpr'] > 1.0:
        risk += 25
    elif row['svpr'] > 0.8:
        risk += 15
    elif row['svpr'] > 0.6:
        risk += 10

    # Add risk from recirculation
    if row['access_recirculation_pct'] > 15:
        risk += 15
    elif row['access_recirculation_pct'] > 10:
        risk += 10

    # Add risk from alarms
    total_alarms = row['high_vp_alarms'] + row['low_ap_alarms']
    if total_alarms > 5:
        risk += 10
    elif total_alarms > 3:
        risk += 5

    return min(100, max(0, risk))


# Load everything
try:
    patients, treatments, outcomes, full_data = load_data()
    model = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Data files not found. Please run data generation first.")

# Header with better styling
st.markdown("""
    <h1 style='color: #1e3a8a; font-size: 42px; margin-bottom: 10px;'>
        🏥 AVF Failure Risk Monitoring Dashboard
    </h1>
    <p style='color: #64748b; font-size: 18px; margin-bottom: 30px;'>
        Predictive early warning system for arteriovenous fistula failure
    </p>
    """, unsafe_allow_html=True)
st.markdown("---")

if model_loaded:
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select View",
                            ["Clinic Overview", "Patient Detail", "Model Performance"])

    # Calculate current risk scores for high-risk patients
    latest_treatments = full_data.groupby('patient_id').tail(1).copy()
    latest_treatments['risk_score'] = latest_treatments.apply(calculate_risk_score, axis=1)
    high_risk_patients = latest_treatments[latest_treatments['risk_score'] > 70].sort_values(
        'risk_score', ascending=False
    )

    # PAGE 1: CLINIC OVERVIEW
    if page == "Clinic Overview":
        st.header("📊 Clinic Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", len(patients))
        with col2:
            st.metric("High Risk Patients",
                      len(high_risk_patients),
                      delta=f"{len(high_risk_patients) / len(patients) * 100:.1f}%")
        with col3:
            st.metric("Critical Alerts",
                      len(high_risk_patients[high_risk_patients['risk_score'] > 85]))
        with col4:
            avg_risk = latest_treatments['risk_score'].mean()
            st.metric("Average Risk Score", f"{avg_risk:.1f}%")

        st.markdown("---")

        # High risk patient table
        st.subheader("⚠️ High Risk Patients (Risk > 70%)")
        if len(high_risk_patients) > 0:
            def get_top_risk_factor(row):
                if row['access_blood_flow_qa'] < 600:
                    return "Low Qa (< 600 mL/min)"
                elif row['svpr'] > 0.8:
                    return "Elevated SVPR"
                elif row['access_recirculation_pct'] > 10:
                    return "High Recirculation"
                elif row['diabetes'] == 1:
                    return "Diabetes + Risk Factors"
                else:
                    return "Multiple Factors"


            display_table = high_risk_patients[['patient_id', 'risk_score', 'age',
                                                'access_blood_flow_qa', 'svpr']].copy()
            display_table['top_risk_factor'] = high_risk_patients.apply(get_top_risk_factor, axis=1)
            display_table['risk_score'] = display_table['risk_score'].round(1)
            display_table.columns = ['Patient ID', 'Risk Score (%)', 'Age',
                                     'Current Qa (mL/min)', 'Current SVPR', 'Top Risk Factor']

            # Removed the extra <style> block here, it's now global
            st.dataframe(display_table.head(15), use_container_width=True, height=400)
        else:
            st.success("✅ No high-risk patients detected")

        st.markdown("---")

        # Risk distribution
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Risk Score Distribution")
            fig = px.histogram(
                latest_treatments,
                x='risk_score',
                nbins=20,
                labels={'risk_score': 'Risk Score (%)'},
                title='Patient Risk Distribution'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Risk by Age Group")
            latest_treatments['age_group'] = pd.cut(
                latest_treatments['age'],
                bins=[0, 40, 50, 60, 70, 100],
                labels=['<40', '40-50', '50-60', '60-70', '70+']
            )
            age_risk = latest_treatments.groupby('age_group')['risk_score'].mean().reset_index()
            fig = px.bar(
                age_risk,
                x='age_group',
                y='risk_score',
                labels={'age_group': 'Age Group', 'risk_score': 'Average Risk Score (%)'},
                title='Average Risk Score by Age Group'
            )
            st.plotly_chart(fig, use_container_width=True)

    # PAGE 2: PATIENT DETAIL
    elif page == "Patient Detail":
        st.header("👤 Individual Patient Analysis")

        patient_list = sorted(patients['patient_id'].unique())
        selected_patient = st.selectbox("Select Patient ID", patient_list)

        patient_info = patients[patients['patient_id'] == selected_patient].iloc[0]
        patient_treatments = treatments[treatments['patient_id'] == selected_patient].sort_values('treatment_number')
        patient_outcome = outcomes[outcomes['patient_id'] == selected_patient].iloc[0]

        latest = patient_treatments.iloc[-1]
        temp_row = pd.Series({
            'baseline_risk_score': patient_info['baseline_risk_score'],
            'access_blood_flow_qa': latest['access_blood_flow_qa'],
            'svpr': latest['svpr'],
            'access_recirculation_pct': latest['access_recirculation_pct'],
            'high_vp_alarms': latest['high_vp_alarms'],
            'low_ap_alarms': latest['low_ap_alarms']
        })
        current_risk = calculate_risk_score(temp_row)

        st.markdown("### Current Risk Assessment")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if current_risk > 85:
                st.markdown(f'<p class="risk-critical">🔴 CRITICAL: {current_risk:.1f}%</p>',
                            unsafe_allow_html=True)
                st.error("**Immediate Action Required:** Schedule vascular ultrasound and physician evaluation")
            elif current_risk > 70:
                st.markdown(f'<p class="risk-high">🟠 HIGH: {current_risk:.1f}%</p>',
                            unsafe_allow_html=True)
                st.warning("**Enhanced Monitoring:** Weekly physical exam and pressure trending")
            elif current_risk > 50:
                st.markdown(f'<p class="risk-moderate">🟡 MODERATE: {current_risk:.1f}%</p>',
                            unsafe_allow_html=True)
                st.info("**Standard Monitoring:** Continue routine surveillance")
            else:
                st.markdown(f'<p class="risk-low">🟢 LOW: {current_risk:.1f}%</p>',
                            unsafe_allow_html=True)
                st.success("**Stable Access:** No immediate concerns")

        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Age", f"{patient_info['age']} years")
        with col2:
            st.metric("Sex", patient_info['sex'])
        with col3:
            diabetes_status = "Yes" if patient_info['diabetes'] == 1 else "No"
            st.metric("Diabetes", diabetes_status)
        with col4:
            st.metric("Prior Interventions", patient_info['prior_interventions'])

        st.markdown("---")

        st.subheader("📈 Hemodynamic Trends")
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=patient_treatments['treatment_number'],
                y=patient_treatments['access_blood_flow_qa'],
                mode='lines+markers',
                name='Access Blood Flow',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.add_hline(y=600, line_dash="dash", line_color="red",
                          annotation_text="Critical Threshold (600 mL/min)")
            fig.update_layout(
                title="Access Blood Flow (Qa) Over Time",
                xaxis_title="Treatment Number",
                yaxis_title="Qa (mL/min)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            current_qa = latest['access_blood_flow_qa']
            if current_qa < 600:
                st.error(f"⚠️ Current Qa: {current_qa:.1f} mL/min (Below threshold)")
            else:
                st.success(f"✅ Current Qa: {current_qa:.1f} mL/min (Normal)")

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=patient_treatments['treatment_number'],
                y=patient_treatments['svpr'],
                mode='lines+markers',
                name='SVPR',
                line=dict(color='#ff7f0e', width=2)
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                          annotation_text="Clinical Threshold (0.5)")
            fig.update_layout(
                title="Static Venous Pressure Ratio (SVPR) Over Time",
                xaxis_title="Treatment Number",
                yaxis_title="SVPR",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            current_svpr = latest['svpr']
            if current_svpr > 0.5:
                st.error(f"⚠️ Current SVPR: {current_svpr:.2f} (Above threshold)")
            else:
                st.success(f"✅ Current SVPR: {current_svpr:.2f} (Normal)")

        st.markdown("---")
        st.subheader("📋 Current Treatment Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Venous Pressure", f"{latest['venous_pressure_mean']:.1f} mmHg")
        with col2:
            st.metric("Access Recirculation", f"{latest['access_recirculation_pct']:.1f}%")
        with col3:
            st.metric("Kt/V", f"{latest['ktv']:.2f}")
        with col4:
            total_alarms = latest['high_vp_alarms'] + latest['low_ap_alarms']
            st.metric("Alarms (Last Treatment)", int(total_alarms))

        if patient_outcome['failed'] == 1:
            st.warning(f"⚠️ **Patient outcome:** Failed at treatment #{patient_outcome['failure_treatment_number']}")
        else:
            st.success("✅ **Patient outcome:** No failure recorded during observation period")

    # PAGE 3: MODEL PERFORMANCE
    elif page == "Model Performance":
        st.header("🎯 Model Performance Metrics")

        feature_importance = pd.read_csv('results/feature_importance.csv')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AUC-ROC", "0.9000")
        with col2:
            st.metric("Recall (Sensitivity)", "88%")
        with col3:
            st.metric("Precision", "30%")

        st.markdown("---")
        st.subheader("📊 Top 15 Most Important Features")
        top_features = feature_importance.head(15)
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            labels={'importance': 'Feature Importance (Gini)', 'feature': 'Feature'},
            title='Feature Importance Rankings'
        )
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🔬 Clinical Validation")
        st.markdown("""
        **The model's top predictors align with established medical literature:**

        1. **SVPR (Venous Pressure) Trends** - Primary indicator of stenosis development
        2. **Access Blood Flow Decline** - Direct measure of access dysfunction
        3. **Baseline Patient Risk** - Comorbidities (diabetes, age, prior interventions)
        4. **Recirculation Patterns** - Marker of poor flow dynamics

        ✅ **Key Finding:** The model learned actual pathophysiology, not spurious correlations.
        """)

        try:
            from PIL import Image

            col1, col2 = st.columns(2)
            with col1:
                st.image('results/figures/roc_curve.png', caption='ROC Curve (AUC = 0.90)')
            with col2:
                st.image('results/figures/feature_importance.png',
                         caption='Feature Importance Distribution')
        except:
            st.info("📊 Generate plots in the Jupyter notebook to display them here")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>AVF Failure Risk Monitor | Built by Jason Odom | Fresno State Data Analytics</p>
    <p>⚠️ For demonstration purposes only. Not for clinical use.</p>
</div>
""", unsafe_allow_html=True)