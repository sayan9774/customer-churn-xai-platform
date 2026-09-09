import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure local directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import (
    generate_synthetic_churn_data,
    prepare_train_test_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from src.model_trainer import (
    train_model,
    evaluate_model,
    train_and_compare_all_models
)
from src.xai_engine import (
    create_shap_explainer,
    compute_shap_values,
    get_local_customer_shap_explanation
)
from src.ai_analyst import generate_rule_based_retention_report


class TestChurnXAIPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Generates sample dataset and prepares preprocessed train/test data."""
        cls.df = generate_synthetic_churn_data(n_samples=200, seed=42)
        (
            cls.X_train,
            cls.X_test,
            cls.y_train,
            cls.y_test,
            cls.preprocessor,
            cls.feature_names,
            cls.X_tr_raw,
            cls.X_te_raw
        ) = prepare_train_test_data(cls.df, random_state=42)
        
    def test_data_generation(self):
        """Test dataset shape and column existence."""
        self.assertEqual(len(self.df), 200)
        self.assertIn("Churn", self.df.columns)
        self.assertIn("Customer_ID", self.df.columns)
        for num_col in NUMERICAL_FEATURES:
            self.assertIn(num_col, self.df.columns)
        for cat_col in CATEGORICAL_FEATURES:
            self.assertIn(cat_col, self.df.columns)

    def test_model_training_and_evaluation(self):
        """Test training XGBoost and checking evaluation metrics."""
        model = train_model("XGBoost", self.X_train, self.y_train)
        metrics = evaluate_model(model, self.X_test, self.y_test)
        
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.5)
        self.assertGreaterEqual(metrics["roc_auc"], 0.5)

    def test_shap_explainer(self):
        """Test SHAP explainer creation and local customer feature attribution."""
        model = train_model("XGBoost", self.X_train, self.y_train)
        explainer = create_shap_explainer(model, self.X_train)
        
        sample_row = self.X_test.iloc[[0]]
        shap_vals, shap_exp = compute_shap_values(explainer, sample_row)
        
        self.assertEqual(shap_vals.shape[1], len(self.feature_names))
        
        raw_cust = self.X_te_raw.iloc[0].to_dict()
        raw_cust["Customer_ID"] = "CUST-TEST"
        
        inc, dec, base_val, shap_sum = get_local_customer_shap_explanation(
            explainer, sample_row, raw_cust
        )
        self.assertIsInstance(inc, list)
        self.assertIsInstance(dec, list)

    def test_ai_analyst_report(self):
        """Test rule-based AI retention report text generation."""
        raw_cust = {
            "Customer_ID": "CUST-1001",
            "Tenure_Months": 3,
            "Monthly_Charges": 95.0,
            "Contract_Type": "Month-to-month",
            "Tech_Support_Tickets": 5,
            "Customer_Satisfaction": 1
        }
        inc_factors = [{"feature": "Contract_Type_Month-to-month", "shap_value": 0.8, "feature_value": 1.0}]
        dec_factors = [{"feature": "Tenure_Months", "shap_value": -0.2, "feature_value": 3.0}]
        
        report = generate_rule_based_retention_report(raw_cust, 0.85, inc_factors, dec_factors)
        
        self.assertIn("CUST-1001", report)
        self.assertIn("CRITICAL HIGH RISK", report)
        self.assertIn("AI Analyst Executive Summary", report)


if __name__ == "__main__":
    unittest.main()
