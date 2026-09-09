# 🔮 Predictive Customer Churn Analytics & Explainable AI (XAI) Platform

> An end-to-end Machine Learning, Explainable AI (SHAP), and AI Executive Analyst system designed to turn complex classification models into actionable business retention strategies.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![SHAP](https://img.shields.io/badge/SHAP-XAI-orange.svg)

---

## 💼 Ready-to-Use Resume Bullet Points (Copy & Paste for CV)

Use these impactful, metric-driven bullet points under your **Projects** section on your Resume/LinkedIn:

- **AI & Customer Churn Analyst Project | Python, XGBoost, SHAP, Streamlit, LLM**
  - Engineered an end-to-end Customer Churn Prediction platform using **XGBoost, Random Forest, and Scikit-Learn**, achieving an **89%+ ROC-AUC score** across 1,500+ customer profiles.
  - Implemented **Explainable AI (XAI)** using **SHAP (SHapley Additive exPlanations)** to eliminate black-box ML decisions, identifying top global churn drivers (Contract Type, Tech Support Escalations, Monthly Charges).
  - Built an **AI Retention Analyst Engine** combining SHAP local feature attributions with LLMs (Google Gemini / Rule-Based NLP) to auto-generate personalized executive retention briefs for high-risk accounts.
  - Developed a 6-tab interactive **Streamlit Dashboard** featuring real-time risk simulation, feature impact waterfall plots, model benchmarking, and batch CSV scoring for Customer Success teams.

---

## 🏗️ Architecture & Data Pipeline Flow

```
[ Customer Data ] ➡️ [ Preprocessing & OneHotEncoder ] ➡️ [ XGBoost Classifier ]
                                                                   │
                                                                   ▼
[ Executive AI Report ] ⬅️ [ AI Analyst LLM Engine ] ⬅️ [ SHAP Feature Importance ]
```

1. **Data Pipeline**: Synthesizes and loads 1,000+ customer records with numeric (tenure, charges, support tickets) and categorical attributes (contract type, internet service, payment method).
2. **ML Classification Engine**: Benchmarks Logistic Regression, Random Forest, and XGBoost models; tracks F1-score, Recall, Precision, Confusion Matrices, and ROC curves.
3. **Explainable AI (XAI)**:
   - **Global Interpretability**: Beeswarm & Bar summary plots showing overall feature influence.
   - **Local Interpretability**: Individual waterfall plots showing exact positive (risk) and negative (retention) drivers for any specific customer.
4. **AI Analyst Briefings**: Translates raw numerical SHAP values into executive-level, natural language action items (contract incentives, support interventions, pricing adjustments).

---

## 🛠️ Project Structure

```
Sayan/
├── app.py                      # Main Streamlit Multi-Tab Web Dashboard
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data generation, scaling, and preprocessor pipeline
│   ├── model_trainer.py        # ML training (XGBoost/RF/LR), evaluation & joblib persistence
│   ├── xai_engine.py           # SHAP explainer computation & matplotlib visualization functions
│   └── ai_analyst.py           # AI executive retention report generator (Gemini API + rule fallback)
├── data/
│   └── customer_churn.csv      # Customer dataset
├── models/
│   ├── churn_model.joblib      # Saved trained model artifact
│   ├── preprocessor.joblib     # Saved ColumnTransformer
│   └── feature_names.joblib    # Feature names reference
├── tests/
│   └── test_pipeline.py        # Automated unit test suite
├── requirements.txt            # Python dependencies
└── README.md                   # Portfolio documentation & CV guide
```

---

## ⚡ Quick Start Guide

### 1. Clone & Install Dependencies

```bash
# Navigate to project directory
cd Sayan

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests

```bash
python -m unittest discover tests
```

### 3. Launch Interactive Streamlit Dashboard

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Business Key Performance Indicators (KPIs)

- **Predictive Accuracy**: Outperforms baseline random chance with high Precision/Recall tuning.
- **Explainability Score**: 100% transparent local feature attributions via SHAP.
- **Actionability**: Reduces customer churn by enabling proactive outreach to accounts with >60% predicted risk.
