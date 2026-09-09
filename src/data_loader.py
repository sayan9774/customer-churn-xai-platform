import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

NUMERICAL_FEATURES = [
    "Age",
    "Tenure_Months",
    "Monthly_Charges",
    "Total_Charges",
    "Tech_Support_Tickets",
    "Customer_Satisfaction",
    "Usage_GB_Per_Month"
]

CATEGORICAL_FEATURES = [
    "Contract_Type",
    "Internet_Service",
    "Payment_Method",
    "Paperless_Billing"
]

def generate_synthetic_churn_data(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic customer dataset for churn analysis.
    The churn probability logic incorporates realistic business rules.
    """
    np.random.seed(seed)
    
    customer_ids = [f"CUST-{1000 + i}" for i in range(n_samples)]
    age = np.random.randint(18, 76, size=n_samples)
    tenure_months = np.random.randint(1, 73, size=n_samples)
    
    contract_type = np.random.choice(
        ["Month-to-month", "One year", "Two year"], 
        size=n_samples, 
        p=[0.55, 0.25, 0.20]
    )
    
    internet_service = np.random.choice(
        ["Fiber optic", "DSL", "No"], 
        size=n_samples, 
        p=[0.45, 0.40, 0.15]
    )
    
    monthly_charges = np.where(
        internet_service == "Fiber optic",
        np.random.uniform(70, 120, size=n_samples),
        np.where(
            internet_service == "DSL",
            np.random.uniform(40, 75, size=n_samples),
            np.random.uniform(20, 35, size=n_samples)
        )
    ).round(2)
    
    total_charges = (tenure_months * monthly_charges + np.random.uniform(-10, 10, size=n_samples)).round(2)
    total_charges = np.clip(total_charges, a_min=20.0, a_max=None)
    
    tech_support_tickets = np.random.poisson(lam=2.0, size=n_samples)
    tech_support_tickets = np.clip(tech_support_tickets, 0, 10)
    
    customer_satisfaction = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.15, 0.20, 0.35, 0.20, 0.10])
    
    usage_gb = np.where(
        internet_service == "No",
        np.random.uniform(0, 15, size=n_samples),
        np.random.uniform(30, 450, size=n_samples)
    ).round(1)
    
    payment_method = np.random.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        size=n_samples,
        p=[0.35, 0.20, 0.25, 0.20]
    )
    
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    
    # Calculate log-odds score for Churn probability
    logit = (
        -0.8
        + 0.035 * (monthly_charges - 65)
        - 0.05 * tenure_months
        + 0.35 * tech_support_tickets
        - 0.6 * (customer_satisfaction - 3)
        + 1.2 * (contract_type == "Month-to-month").astype(int)
        - 0.8 * (contract_type == "Two year").astype(int)
        + 0.6 * (internet_service == "Fiber optic").astype(int)
        + 0.5 * (payment_method == "Electronic Check").astype(int)
    )
    
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < prob_churn).astype(int)
    
    df = pd.DataFrame({
        "Customer_ID": customer_ids,
        "Age": age,
        "Tenure_Months": tenure_months,
        "Contract_Type": contract_type,
        "Internet_Service": internet_service,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges,
        "Tech_Support_Tickets": tech_support_tickets,
        "Customer_Satisfaction": customer_satisfaction,
        "Usage_GB_Per_Month": usage_gb,
        "Payment_Method": payment_method,
        "Paperless_Billing": paperless_billing,
        "Churn": churn
    })
    
    return df

def load_or_create_data(filepath: str = "data/customer_churn.csv") -> pd.DataFrame:
    """Loads dataset from file if exists, otherwise generates and saves synthetic data."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = generate_synthetic_churn_data()
        df.to_csv(filepath, index=False)
    return df

def build_preprocessor() -> ColumnTransformer:
    """Builds a ColumnTransformer for numeric scaling and categorical encoding."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def prepare_train_test_data(df: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2, random_state: int = 42):
    """
    Preprocesses data and splits into train/test sets.
    Returns: X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names, X_train_raw, X_test_raw
    """
    feature_cols = [c for c in df.columns if c not in ["Customer_ID", target_col]]
    X = df[feature_cols]
    y = df[target_col]
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    
    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERICAL_FEATURES + encoded_cat_names
    
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names, index=X_test_raw.index)
    
    return X_train_df, X_test_df, y_train, y_test, preprocessor, feature_names, X_train_raw, X_test_raw
