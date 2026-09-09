import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

HARDCODED_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def generate_rule_based_retention_report(
    customer_raw: Dict[str, Any],
    churn_prob: float,
    top_risk_factors: List[Dict[str, Any]],
    top_retention_factors: List[Dict[str, Any]]
) -> str:
    """
    Generates a structured executive AI retention strategy report
    without requiring an external API key.
    """
    customer_id = customer_raw.get("Customer_ID", "CUST-UNKNOWN")
    tenure = customer_raw.get("Tenure_Months", "N/A")
    charges = customer_raw.get("Monthly_Charges", "N/A")
    contract = customer_raw.get("Contract_Type", "N/A")
    tickets = customer_raw.get("Tech_Support_Tickets", 0)
    satisfaction = customer_raw.get("Customer_Satisfaction", 3)
    
    risk_level = "CRITICAL HIGH RISK" if churn_prob >= 0.7 else ("MODERATE RISK" if churn_prob >= 0.4 else "LOW RISK / HEALTHY")
    
    # Format risk factor strings
    risk_strs = []
    for f in top_risk_factors:
        feat = f["feature"]
        val = f["shap_value"]
        if "Contract_Type" in feat:
            risk_strs.append(f"Short-term **{contract}** contract structure (+{val:.2f} impact)")
        elif "Monthly_Charges" in feat:
            risk_strs.append(f"High monthly billing of **${charges}/mo** (+{val:.2f} impact)")
        elif "Tech_Support_Tickets" in feat:
            risk_strs.append(f"Frequent support escalations (**{tickets} tickets**) (+{val:.2f} impact)")
        elif "Customer_Satisfaction" in feat:
            risk_strs.append(f"Low satisfaction rating (**{satisfaction}/5**) (+{val:.2f} impact)")
        elif "Tenure" in feat:
            risk_strs.append(f"Low account tenure (**{tenure} months**) (+{val:.2f} impact)")
        else:
            risk_strs.append(f"Feature `{feat}` (+{val:.2f} impact)")
            
    retention_strs = []
    for f in top_retention_factors:
        feat = f["feature"]
        val = abs(f["shap_value"])
        if "Tenure" in feat:
            retention_strs.append(f"Established tenure (**{tenure} months**) (-{val:.2f} risk reduction)")
        elif "Contract" in feat:
            retention_strs.append(f"Long-term commitment (**{contract}**) (-{val:.2f} risk reduction)")
        elif "Satisfaction" in feat:
            retention_strs.append(f"Strong feedback score (**{satisfaction}/5**) (-{val:.2f} risk reduction)")
        else:
            retention_strs.append(f"Favorable indicator `{feat}` (-{val:.2f} risk reduction)")

    report = f"""
### 📊 AI Analyst Executive Summary
- **Customer ID**: `{customer_id}`
- **Predicted Churn Risk**: `{churn_prob * 100:.1f}%` (**{risk_level}**)
- **Contract Type**: {contract} | **Tenure**: {tenure} months | **Monthly Bill**: ${charges}

---

#### 🚨 Primary Churn Drivers (Identified by SHAP Explainable AI)
{chr(10).join(f"- {r}" for r in risk_strs) if risk_strs else "- No major risk drivers detected."}

#### ✅ Key Stabilization Drivers
{chr(10).join(f"- {s}" for s in retention_strs) if retention_strs else "- No strong stabilization factors detected."}

---

#### 💡 Prescriptive Action Plan & Strategic Retention Steps
1. **Contract Incentive**: {"Offer a 15% discount on an annual contract renewal to transition off month-to-month billing." if contract == "Month-to-month" else "Lock in current billing rates for another year with a loyalty perk."}
2. **Support Resolution**: {"Assign a dedicated senior technical specialist to address unresolved open tickets." if tickets > 1 else "Proactively touch base via Customer Success team to ensure onboarding satisfaction."}
3. **Price Sensitivity Management**: {"Review feature utilization to offer a right-sized tier or bundle discount." if (float(charges) if isinstance(charges, (int, float)) else 80) > 70 else "Maintain current value proposition."}
4. **Immediate Outreach Protocol**: {"Trigger high-priority phone call from Retention Team within 24 hours." if churn_prob >= 0.6 else "Include in targeted quarterly retention email campaign."}
"""
    return report.strip()

def generate_ai_retention_report(
    customer_raw: Dict[str, Any],
    churn_prob: float,
    top_risk_factors: List[Dict[str, Any]],
    top_retention_factors: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> str:
    """
    Generates an executive AI retention report using Gemini API if API key is provided,
    otherwise gracefully falls back to the intelligent rule-based AI engine.
    """
    resolved_api_key = api_key if (api_key and api_key.strip()) else os.getenv("GEMINI_API_KEY", HARDCODED_GEMINI_API_KEY)
    if resolved_api_key and resolved_api_key.strip():
        try:
            from google import genai
            client = genai.Client(api_key=resolved_api_key.strip())
            
            prompt = f"""
You are a Lead AI & Business Analyst advising executive leadership.
Analyze the following customer churn prediction data explained by SHAP Explainable AI:

Customer Profile:
{customer_raw}

Model Churn Probability: {churn_prob * 100:.1f}%

Top SHAP Features Increasing Churn Risk:
{top_risk_factors}

Top SHAP Features Reducing Churn Risk:
{top_retention_factors}

Provide a concise, highly professional executive retention briefing with:
1. Executive Risk Summary & Risk Rating
2. Root Cause Analysis (Explaining SHAP drivers in business terms)
3. 3-4 Actionable, Prioritized Retention Recommendations for Customer Success & Sales Teams.
Format in clean Markdown with clear headings and bullet points.
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            if response and response.text:
                return response.text
        except Exception as e:
            # Fall back to rule-based engine on API error
            fallback_note = f"> *Note: Gemini API call unfulfilled ({str(e)}). Displaying built-in AI Analyst Report.*\n\n"
            return fallback_note + generate_rule_based_retention_report(
                customer_raw, churn_prob, top_risk_factors, top_retention_factors
            )
            
    return generate_rule_based_retention_report(
        customer_raw, churn_prob, top_risk_factors, top_retention_factors
    )
