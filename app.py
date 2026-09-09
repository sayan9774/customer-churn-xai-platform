import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from dotenv import load_dotenv

load_dotenv()

# Ensure local src directory is on path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import (
    load_or_create_data,
    prepare_train_test_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.model_trainer import (
    train_model,
    evaluate_model,
    train_and_compare_all_models,
    save_model_artifacts
)
from src.xai_engine import (
    create_shap_explainer,
    compute_shap_values,
    get_global_feature_importance,
    get_local_customer_shap_explanation,
    plot_shap_summary_bar,
    plot_shap_beeswarm,
    plot_shap_waterfall_single
)
from src.ai_analyst import generate_ai_retention_report

# Page Configuration
st.set_page_config(
    page_title="AI Analyst | Customer Churn & XAI Platform",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0px 0px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_cached_data():
    return load_or_create_data("data/customer_churn.csv")

@st.cache_resource
def build_and_cache_models():
    df = get_cached_data()
    X_train, X_test, y_train, y_test, preprocessor, feature_names, X_tr_raw, X_te_raw = prepare_train_test_data(df)
    models, evals, best_name = train_and_compare_all_models(X_train, y_train, X_test, y_test)
    
    # Save artifacts
    save_model_artifacts(models[best_name], preprocessor, feature_names)
    
    return {
        "df": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "models": models,
        "evaluations": evals,
        "best_model_name": best_name,
        "X_tr_raw": X_tr_raw,
        "X_te_raw": X_te_raw
    }

def main():
    st.markdown('<div class="main-header">🔮 Customer Churn Analytics & Explainable AI (XAI) Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated ML classification (XGBoost) combined with SHAP feature interpretability & AI executive retention strategies</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.title("⚙️ Control Panel")
    
    pipeline = build_and_cache_models()
    df = pipeline["df"]
    models = pipeline["models"]
    evaluations = pipeline["evaluations"]
    
    selected_model_name = st.sidebar.selectbox(
        "Select Active ML Model",
        options=list(models.keys()),
        index=list(models.keys()).index(pipeline["best_model_name"])
    )
    active_model = models[selected_model_name]
    
    st.sidebar.markdown("---")
    st.sidebar.caption("🔒 Backend Secured AI Engine Active")
    
    # Pre-compute SHAP explainer for active model
    explainer = create_shap_explainer(active_model, pipeline["X_train"])
    
    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Overview",
        "🔍 EDA & Metrics",
        "🤖 Model Benchmarking",
        "⚡ Single Customer XAI",
        "💡 AI Retention Analyst",
        "📁 Batch Prediction"
    ])

    # ----------------------------------------------------
    # TAB 1: EXECUTIVE OVERVIEW
    # ----------------------------------------------------
    with tab1:
        st.subheader("📌 Key Business Indicators")
        
        total_cust = len(df)
        churned_cust = df["Churn"].sum()
        churn_rate = (churned_cust / total_cust) * 100
        avg_charge = df["Monthly_Charges"].mean()
        avg_tenure = df["Tenure_Months"].mean()
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Customers", f"{total_cust:,}")
        c2.metric("Churned Customers", f"{churned_cust:,}")
        c3.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        c4.metric("Avg Monthly Bill", f"${avg_charge:.2f}")
        c5.metric("Avg Account Tenure", f"{avg_tenure:.1f} mos")
        
        st.markdown("---")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### Customer Churn Distribution")
            fig_churn = px.pie(
                df, 
                names=df["Churn"].map({0: "Retained", 1: "Churned"}), 
                hole=0.4,
                color_discrete_sequence=["#10B981", "#EF4444"]
            )
            st.plotly_chart(fig_churn, use_container_width=True)
            
        with col_right:
            st.markdown("#### Churn Rate by Contract Type")
            contract_churn = df.groupby("Contract_Type")["Churn"].mean().reset_index()
            contract_churn["Churn_Rate_%"] = contract_churn["Churn"] * 100
            fig_contract = px.bar(
                contract_churn,
                x="Contract_Type",
                y="Churn_Rate_%",
                color="Contract_Type",
                text_auto=".1f",
                labels={"Churn_Rate_%": "Churn Rate (%)"},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_contract, use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
    # ----------------------------------------------------
    with tab2:
        st.subheader("🔍 Exploratory Data Analysis")
        
        feature_to_plot = st.selectbox(
            "Select Feature to Analyze vs Churn Status",
            options=NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
            index=2
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if feature_to_plot in NUMERICAL_FEATURES:
                fig_dist = px.histogram(
                    df,
                    x=feature_to_plot,
                    color=df["Churn"].map({0: "Retained", 1: "Churned"}),
                    barmode="overlay",
                    marginal="box",
                    title=f"Distribution of {feature_to_plot} by Churn",
                    color_discrete_sequence=["#10B981", "#EF4444"]
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                cat_df = df.groupby([feature_to_plot, "Churn"]).size().reset_index(name="Count")
                cat_df["Status"] = cat_df["Churn"].map({0: "Retained", 1: "Churned"})
                fig_cat = px.bar(
                    cat_df,
                    x=feature_to_plot,
                    y="Count",
                    color="Status",
                    barmode="group",
                    title=f"{feature_to_plot} Breakdown by Churn Status",
                    color_discrete_sequence=["#10B981", "#EF4444"]
                )
                st.plotly_chart(fig_cat, use_container_width=True)
                
        with col2:
            st.markdown("#### Numerical Feature Correlation Matrix")
            corr = df[NUMERICAL_FEATURES + ["Churn"]].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Feature Correlations"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: MODEL BENCHMARKING & GLOBAL SHAP
    # ----------------------------------------------------
    with tab3:
        st.subheader("🤖 Machine Learning Model Benchmarking")
        
        # Summary Table
        metrics_list = []
        for name, ev in evaluations.items():
            metrics_list.append({
                "Model Name": name,
                "Accuracy": f"{ev['accuracy']*100:.2f}%",
                "Precision": f"{ev['precision']*100:.2f}%",
                "Recall": f"{ev['recall']*100:.2f}%",
                "F1 Score": f"{ev['f1_score']*100:.2f}%",
                "ROC-AUC": f"{ev['roc_auc']:.3f}"
            })
        st.dataframe(pd.DataFrame(metrics_list), use_container_width=True)
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown(f"#### Confusion Matrix ({selected_model_name})")
            cm = np.array(evaluations[selected_model_name]["confusion_matrix"])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted Label", y="True Label"),
                x=["Retained (0)", "Churned (1)"],
                y=["Retained (0)", "Churned (1)"],
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_m2:
            st.markdown("#### ROC Curve Comparison")
            fig_roc = go.Figure()
            for name, ev in evaluations.items():
                fig_roc.add_trace(go.Scatter(
                    x=ev["roc_curve"]["fpr"],
                    y=ev["roc_curve"]["tpr"],
                    mode="lines",
                    name=f"{name} (AUC={ev['roc_auc']:.3f})"
                ))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="Random Chance"))
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)
            
        st.markdown("---")
        st.subheader("🌍 Global Feature Importance (SHAP Explainable AI)")
        
        with st.spinner("Computing global SHAP values..."):
            shap_vals, shap_exp = compute_shap_values(explainer, pipeline["X_test"])
            
        c_shap1, c_shap2 = st.columns(2)
        with c_shap1:
            st.markdown("#### Global Feature Importance (Mean |SHAP| Value)")
            fig_bar = plot_shap_summary_bar(shap_vals, pipeline["X_test"])
            st.pyplot(fig_bar)
        with c_shap2:
            st.markdown("#### SHAP Beeswarm Distribution Plot")
            fig_bee = plot_shap_beeswarm(shap_exp)
            st.pyplot(fig_bee)

    # ----------------------------------------------------
    # TAB 4: SINGLE CUSTOMER XAI & RISK SIMULATOR
    # ----------------------------------------------------
    with tab4:
        st.subheader("⚡ Single Customer Churn Risk & XAI Simulator")
        st.caption("Adjust customer attributes below to simulate model prediction and view feature-level SHAP explanation.")
        
        col_inp1, col_inp2, col_inp3 = st.columns(3)
        
        with col_inp1:
            sim_tenure = st.slider("Tenure (Months)", 1, 72, 6)
            sim_monthly = st.slider("Monthly Charges ($)", 20.0, 120.0, 85.0)
            sim_tickets = st.slider("Tech Support Tickets", 0, 10, 4)
            sim_age = st.slider("Customer Age", 18, 85, 42)
            
        with col_inp2:
            sim_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
            sim_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0)
            sim_satisfaction = st.slider("Customer Satisfaction (1-5)", 1, 5, 2)
            
        with col_inp3:
            sim_payment = st.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"], index=0)
            sim_paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
            sim_usage = st.number_input("Usage (GB/month)", min_value=0.0, max_value=600.0, value=250.0)
            
        sim_total = round(sim_tenure * sim_monthly, 2)
        
        raw_cust_dict = {
            "Customer_ID": "CUST-SIMULATED",
            "Age": sim_age,
            "Tenure_Months": sim_tenure,
            "Contract_Type": sim_contract,
            "Internet_Service": sim_internet,
            "Monthly_Charges": sim_monthly,
            "Total_Charges": sim_total,
            "Tech_Support_Tickets": sim_tickets,
            "Customer_Satisfaction": sim_satisfaction,
            "Usage_GB_Per_Month": sim_usage,
            "Payment_Method": sim_payment,
            "Paperless_Billing": sim_paperless
        }
        
        cust_df_raw = pd.DataFrame([raw_cust_dict]).drop(columns=["Customer_ID"])
        cust_proc = pipeline["preprocessor"].transform(cust_df_raw)
        cust_proc_df = pd.DataFrame(cust_proc, columns=pipeline["feature_names"])
        
        churn_prob = float(active_model.predict_proba(cust_proc_df)[0, 1])
        
        st.markdown("---")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.markdown("#### Risk Prediction Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(churn_prob * 100, 1),
                title={'text': "Predicted Churn Risk (%)"},
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1E293B"},
                    'steps': [
                        {'range': [0, 35], 'color': "#D1FAE5"},
                        {'range': [35, 65], 'color': "#FEF3C7"},
                        {'range': [65, 100], 'color': "#FEE2E2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_res2:
            st.markdown("#### SHAP Local Waterfall Explanation")
            fig_waterfall = plot_shap_waterfall_single(explainer, cust_proc_df)
            st.pyplot(fig_waterfall)
            
        # Store in session state for Tab 5
        st.session_state["sim_raw"] = raw_cust_dict
        st.session_state["sim_prob"] = churn_prob
        st.session_state["sim_proc_df"] = cust_proc_df

    # ----------------------------------------------------
    # TAB 5: AI RETENTION ANALYST
    # ----------------------------------------------------
    with tab5:
        st.subheader("💡 AI Analyst Executive Retention Briefing")
        
        sim_raw = st.session_state.get("sim_raw", raw_cust_dict)
        sim_prob = st.session_state.get("sim_prob", churn_prob)
        sim_proc_df = st.session_state.get("sim_proc_df", cust_proc_df)
        
        inc_factors, dec_factors, b_val, s_sum = get_local_customer_shap_explanation(
            explainer, sim_proc_df, sim_raw
        )
        
        if st.button("🚀 Generate Executive AI Retention Report", type="primary"):
            with st.spinner("Analyzing SHAP risk drivers & synthesizing strategic recommendations..."):
                report_md = generate_ai_retention_report(
                    sim_raw, sim_prob, inc_factors, dec_factors
                )
                st.session_state["current_report"] = report_md
                
        if "current_report" in st.session_state:
            st.markdown(st.session_state["current_report"])
        else:
            st.info("Click the button above to generate a tailored executive retention briefing for the simulated customer profile.")

    # ----------------------------------------------------
    # TAB 6: BATCH PREDICTION & EXPORT
    # ----------------------------------------------------
    with tab6:
        st.subheader("📁 Batch Customer Prediction & Risk Scoring")
        st.markdown("Upload a CSV containing customer records to predict churn probabilities and export results.")
        
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:", batch_df.head())
            
            if st.button("Run Batch Scoring"):
                with st.spinner("Processing batch predictions..."):
                    req_cols = [c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
                    missing = [c for c in req_cols if c not in batch_df.columns]
                    
                    if missing:
                        st.error(f"Missing required columns in uploaded CSV: {missing}")
                    else:
                        batch_proc = pipeline["preprocessor"].transform(batch_df[req_cols])
                        batch_proc_df = pd.DataFrame(batch_proc, columns=pipeline["feature_names"])
                        probs = active_model.predict_proba(batch_proc_df)[:, 1]
                        
                        out_df = batch_df.copy()
                        out_df["Predicted_Churn_Probability"] = np.round(probs, 4)
                        out_df["Risk_Category"] = np.where(probs >= 0.65, "High", np.where(probs >= 0.35, "Medium", "Low"))
                        
                        st.success(f"Batch scoring completed for {len(out_df)} records!")
                        st.dataframe(out_df, use_container_width=True)
                        
                        csv_data = out_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Scored Results CSV",
                            data=csv_data,
                            file_name="churn_predictions_scored.csv",
                            mime="text/csv"
                        )
        else:
            st.markdown("#### Need a sample template CSV?")
            sample_df = df.head(10).copy()
            if "Churn" in sample_df.columns:
                sample_df = sample_df.drop(columns=["Churn"])
            csv_sample = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Sample Input Template CSV",
                data=csv_sample,
                file_name="sample_churn_batch_template.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
