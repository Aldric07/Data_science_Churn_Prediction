import os
import pandas as pd
import cloudpickle
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# ----------------------------
# Model loading
# ----------------------------
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return cloudpickle.load(f)

# Charge TOUS les modèles au démarrage (une seule fois)
MODELS = {
    "logreg": load_model("models/logreg.pkl"),
    "random_forest": load_model("models/random_forest.pkl"),
    "naive_bayes": load_model("models/naive_bayes.pkl"),
}

app = FastAPI(title="Telecom Churn - Prediction API", version="1.0")


# ----------------------------
# 1) Pydantic schema (snake_case)
# ----------------------------
class ChurnFeatures(BaseModel):
    Gender: str = Field(..., examples=["Male", "Female"])
    Age: int = Field(..., ge=0, le=120)
    Married: str = Field(..., examples=["Yes", "No"])

    Number_of_Dependents: int = Field(..., ge=0)
    Number_of_Referrals: int = Field(..., ge=0)
    Tenure_in_Months: int = Field(..., ge=0)

    City: str = Field(..., examples=["Frazier Park", "San Diego"])
    Zip_Code: int = Field(..., ge=0)
    Latitude: float
    Longitude: float

    Offer: str = Field(..., examples=["No Offer", "Offer A", "Offer B"])
    Phone_Service: str = Field(..., examples=["Yes", "No"])
    Multiple_Lines: str = Field(..., examples=["Yes", "No", "No Phone Service"])

    Internet_Service: str = Field(..., examples=["Yes", "No"])
    Internet_Type: str = Field(..., examples=["No Internet Service", "DSL", "Fiber Optic", "Cable"])
    Avg_Monthly_GB_Download: float = Field(..., ge=0)

    Online_Security: str = Field(..., examples=["Yes", "No", "No Internet Service"])
    Online_Backup: str = Field(..., examples=["Yes", "No", "No Internet Service"])
    Device_Protection_Plan: str = Field(..., examples=["Yes", "No", "No Internet Service"])
    Premium_Tech_Support: str = Field(..., examples=["Yes", "No", "No Internet Service"])
    Streaming_TV: str = Field(..., examples=["Yes", "No", "No Internet Service"])
    Streaming_Movies: str = Field(..., examples=["Yes", "No", "No Internet Service"])
    Streaming_Music: str = Field(..., examples=["Yes", "No", "No Internet Service"])
    Unlimited_Data: str = Field(..., examples=["Yes", "No", "No Internet Service"])

    Contract: str = Field(..., examples=["Month-to-Month", "One Year", "Two Year"])
    Paperless_Billing: str = Field(..., examples=["Yes", "No"])
    Payment_Method: str = Field(..., examples=["Bank Withdrawal", "Credit Card", "Mailed Check"])

    Monthly_Charge: float = Field(..., ge=0)
    Total_Charges: float = Field(..., ge=0)
    Total_Refunds: float = Field(..., ge=0)
    Total_Extra_Data_Charges: float = Field(..., ge=0)
    Total_Long_Distance_Charges: float = Field(..., ge=0)
    Total_Revenue: float = Field(..., ge=0)

    Avg_Monthly_Long_Distance_Charges: float = Field(..., ge=0)


# ----------------------------
# 2) Mapping snake_case -> dataset column names (with spaces)
# ----------------------------
SPACE_COLS_MAP = {
    "Number_of_Dependents": "Number of Dependents",
    "Number_of_Referrals": "Number of Referrals",
    "Tenure_in_Months": "Tenure in Months",
    "Zip_Code": "Zip Code",
    "Phone_Service": "Phone Service",
    "Multiple_Lines": "Multiple Lines",
    "Internet_Service": "Internet Service",
    "Internet_Type": "Internet Type",
    "Avg_Monthly_GB_Download": "Avg Monthly GB Download",
    "Online_Security": "Online Security",
    "Online_Backup": "Online Backup",
    "Device_Protection_Plan": "Device Protection Plan",
    "Premium_Tech_Support": "Premium Tech Support",
    "Streaming_TV": "Streaming TV",
    "Streaming_Movies": "Streaming Movies",
    "Streaming_Music": "Streaming Music",
    "Unlimited_Data": "Unlimited Data",
    "Paperless_Billing": "Paperless Billing",
    "Payment_Method": "Payment Method",
    "Monthly_Charge": "Monthly Charge",
    "Total_Charges": "Total Charges",
    "Total_Refunds": "Total Refunds",
    "Total_Extra_Data_Charges": "Total Extra Data Charges",
    "Total_Long_Distance_Charges": "Total Long Distance Charges",
    "Total_Revenue": "Total Revenue",
    "Avg_Monthly_Long_Distance_Charges": "Avg Monthly Long Distance Charges",
}

EXPECTED_COLS = [
    "Gender", "Age", "Married",
    "Number of Dependents", "City", "Zip Code", "Latitude", "Longitude",
    "Number of Referrals", "Tenure in Months", "Offer",
    "Phone Service", "Avg Monthly Long Distance Charges", "Multiple Lines",
    "Internet Service", "Internet Type", "Avg Monthly GB Download",
    "Online Security", "Online Backup", "Device Protection Plan", "Premium Tech Support",
    "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data",
    "Contract", "Paperless Billing", "Payment Method",
    "Monthly Charge", "Total Charges", "Total Refunds",
    "Total Extra Data Charges", "Total Long Distance Charges", "Total Revenue",
]

DEFAULTS = {
    "City": "Unknown",
    "Offer": "No Offer",
    "Multiple Lines": "No Phone Service",
    "Internet Type": "No Internet Service",
    "Online Security": "No Internet Service",
    "Online Backup": "No Internet Service",
    "Device Protection Plan": "No Internet Service",
    "Premium Tech Support": "No Internet Service",
    "Streaming TV": "No Internet Service",
    "Streaming Movies": "No Internet Service",
    "Streaming Music": "No Internet Service",
    "Unlimited Data": "No Internet Service",
    "Avg Monthly GB Download": 0.0,
    "Avg Monthly Long Distance Charges": 0.0,
    "Total Refunds": 0.0,
    "Total Extra Data Charges": 0.0,
    "Total Long Distance Charges": 0.0,
}


def build_feature_row(payload: ChurnFeatures) -> pd.DataFrame:
    """Transform payload into a DataFrame with the exact columns expected by the trained pipelines."""
    data = payload.model_dump()
    X = pd.DataFrame([data]).rename(columns=SPACE_COLS_MAP)

    row = {col: DEFAULTS.get(col, 0) for col in EXPECTED_COLS}
    row.update(X.iloc[0].to_dict())

    return pd.DataFrame([row], columns=EXPECTED_COLS)


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "available_models": list(MODELS.keys()),
        "expected_cols_count": len(EXPECTED_COLS)
    }


@app.get("/models")
def list_models():
    return {"available_models": list(MODELS.keys())}


@app.post("/predict")
def predict(
    payload: ChurnFeatures,
    model_name: str = Query("logreg", description="Model to use: logreg, random_forest, naive_bayes"),
):
    try:
        if model_name not in MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model_name}'. Choose one of: {list(MODELS.keys())}",
            )

        X_final = build_feature_row(payload)
        mdl = MODELS[model_name]

        y_pred = int(mdl.predict(X_final)[0])

        proba = None
        if hasattr(mdl, "predict_proba"):
            proba = float(mdl.predict_proba(X_final)[:, 1][0])

        return {
            "model": model_name,
            "churn_prediction": y_pred,
            "churn_probability": proba,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

