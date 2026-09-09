import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List

# Ensure user site-packages is included in path
user_site = os.path.expanduser(r"~\AppData\Roaming\Python\Python313\site-packages")
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

def create_shap_explainer(model: Any, X_background: pd.DataFrame = None) -> Any:
    """
    Creates an appropriate SHAP explainer based on model architecture.
    """
    if not HAS_SHAP:
        return None
        
    model_name = type(model).__name__
    
    if "XGB" in model_name or "RandomForest" in model_name or "Tree" in model_name:
        explainer = shap.TreeExplainer(model)
    else:
        if X_background is not None:
            explainer = shap.Explainer(model.predict_proba, X_background)
        else:
            explainer = shap.Explainer(model)
            
    return explainer

def compute_shap_values(explainer: Any, X: pd.DataFrame) -> Tuple[np.ndarray, Any]:
    """
    Computes SHAP values for dataset X.
    Handles binary classification output indexing differences across SHAP versions.
    """
    shap_explanation = explainer(X)
    shap_vals = shap_explanation.values
    
    # If 3D array (samples, features, classes), pick class 1 (churn = 1)
    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]
        shap_explanation = shap_explanation[:, :, 1]
        
    return shap_vals, shap_explanation

def get_global_feature_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Computes mean absolute SHAP value for each feature across all samples.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Importance": mean_abs_shap
    }).sort_values(by="SHAP_Importance", ascending=False).reset_index(drop=True)
    return df_imp

def get_local_customer_shap_explanation(
    explainer: Any,
    customer_processed_row: pd.DataFrame,
    customer_raw_dict: dict,
    top_n: int = 5
) -> Tuple[List[dict], List[dict], float, float]:
    """
    Explains prediction for a single customer.
    Returns:
      top_increasing_factors: Features pushing churn probability HIGHER (Risk factors)
      top_decreasing_factors: Features pulling churn probability LOWER (Retention factors)
      base_value: Model expectation/baseline logit or probability
      shap_sum: Sum of SHAP values for this instance
    """
    shap_explanation = explainer(customer_processed_row)
    vals = shap_explanation.values
    
    if len(vals.shape) == 2:
        vals = vals[0]
    elif len(vals.shape) == 3:
        vals = vals[0, :, 1]
        
    feature_names = customer_processed_row.columns.tolist()
    feature_values = customer_processed_row.iloc[0].values
    
    factors = []
    for name, val, feat_val in zip(feature_names, vals, feature_values):
        factors.append({
            "feature": name,
            "shap_value": float(val),
            "feature_value": float(feat_val),
            "raw_feature_display": f"{name} = {customer_raw_dict.get(name, feat_val)}"
        })
        
    # Sort by SHAP value magnitude
    increasing = sorted([f for f in factors if f["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:top_n]
    decreasing = sorted([f for f in factors if f["shap_value"] < 0], key=lambda x: x["shap_value"])[:top_n]
    
    bv = shap_explanation.base_values
    if hasattr(bv, "flatten"):
        bv_flat = bv.flatten()
        base_val = float(bv_flat[-1]) if len(bv_flat) > 1 else float(bv_flat[0])
    elif isinstance(bv, (list, tuple)):
        base_val = float(bv[-1])
    else:
        base_val = float(bv)
        
    shap_sum = float(vals.sum())
    
    return increasing, decreasing, base_val, shap_sum

def plot_shap_summary_bar(shap_values: np.ndarray, X: pd.DataFrame, max_display: int = 10) -> plt.Figure:
    """Renders SHAP global feature importance bar plot."""
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    return fig

def plot_shap_beeswarm(shap_explanation: Any, max_display: int = 10) -> plt.Figure:
    """Renders SHAP summary beeswarm plot."""
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.beeswarm(shap_explanation, max_display=max_display, show=False)
    plt.tight_layout()
    return fig

def plot_shap_waterfall_single(explainer: Any, customer_row: pd.DataFrame) -> plt.Figure:
    """Renders SHAP waterfall plot for a single customer."""
    shap_exp = explainer(customer_row)
    if len(shap_exp.values.shape) == 3:
        shap_exp = shap_exp[:, :, 1]
        
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap_exp[0], max_display=8, show=False)
    plt.tight_layout()
    return fig
