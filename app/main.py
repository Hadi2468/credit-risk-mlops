from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Credit Risk API")

class PredictRequest(BaseModel):
    user_id: str = Field(min_length=1)
    avg_payment_delay: float = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    default_probability = min(0.95, req.avg_payment_delay / 20)
    return {
        "user_id": req.user_id,
        "default_probability": round(default_probability, 2),
    }