import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

def train_model(model_type: str, X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> Any:
    """
    Instantiates and trains the specified classification model.
    Supported model_type values: 'XGBoost', 'Random Forest', 'Logistic Regression'.
    """
    model_type = model_type.strip()
    if model_type == "XGBoost":
        if HAS_XGBOOST:
            model = XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                eval_metric="logloss"
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                random_state=random_state
            )
    elif model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=4,
            random_state=random_state
        )
    elif model_type == "Logistic Regression":
        model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported model type: '{model_type}'. Choose from 'XGBoost', 'Random Forest', 'Logistic Regression'.")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluates a trained classifier model and returns performance metrics.
    """
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else 0.0
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist()
    }

def train_and_compare_all_models(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], str]:
    """
    Trains XGBoost, Random Forest, and Logistic Regression models.
    Returns: models_dict, evaluation_results, best_model_name
    """
    model_types = ["XGBoost", "Random Forest", "Logistic Regression"]
    models = {}
    evaluations = {}
    best_model_name = "XGBoost"
    best_f1 = -1.0
    
    for m_type in model_types:
        trained_m = train_model(m_type, X_train, y_train)
        eval_res = evaluate_model(trained_m, X_test, y_test)
        models[m_type] = trained_m
        evaluations[m_type] = eval_res
        
        if eval_res["f1_score"] > best_f1:
            best_f1 = eval_res["f1_score"]
            best_model_name = m_type
            
    return models, evaluations, best_model_name

def save_model_artifacts(model: Any, preprocessor: Any, feature_names: list, model_dir: str = "models"):
    """Saves model and preprocessor artifacts to disk."""
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "churn_model.joblib"))
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))
    joblib.dump(feature_names, os.path.join(model_dir, "feature_names.joblib"))

def load_model_artifacts(model_dir: str = "models") -> Tuple[Any, Any, list]:
    """Loads saved model artifacts from disk if available."""
    model_path = os.path.join(model_dir, "churn_model.joblib")
    prep_path = os.path.join(model_dir, "preprocessor.joblib")
    feat_path = os.path.join(model_dir, "feature_names.joblib")
    
    if os.path.exists(model_path) and os.path.exists(prep_path) and os.path.exists(feat_path):
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        feature_names = joblib.load(feat_path)
        return model, preprocessor, feature_names
    return None, None, None
