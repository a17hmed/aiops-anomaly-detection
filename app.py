from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Self-Healing Cloud Infrastructure (AIOps)",
    version="1.0"
)

# Load artifacts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("workload_type_encoder.pkl")

FEATURE_COLUMNS = [
    "CPU_Usage",
    "Memory_Usage",
    "Disk_IO",
    "Network_IO",
    "Workload_Type_Encoded"
]

class CloudMetrics(BaseModel):
    CPU_Usage: float
    Memory_Usage: float
    Disk_IO: float
    Network_IO: float
    Workload_Type: str

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: CloudMetrics):
    try:
        workload_encoded = encoder.transform([data.Workload_Type])[0]

        x = pd.DataFrame([{
            "CPU_Usage": data.CPU_Usage,
            "Memory_Usage": data.Memory_Usage,
            "Disk_IO": data.Disk_IO,
            "Network_IO": data.Network_IO,
            "Workload_Type_Encoded": workload_encoded
        }])[FEATURE_COLUMNS]

        x_scaled = scaler.transform(x)
        pred = model.predict(x_scaled)[0]

        anomaly = bool(pred == 1)
        action = "restart_container_or_scale_service" if anomaly else "system_healthy"

        return {
            "anomaly": anomaly,
            "recommended_action": action
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))