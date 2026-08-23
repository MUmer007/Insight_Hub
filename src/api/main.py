"""
main.py

Phase 5 - Serving Layer
A FastAPI backend exposing all three InsightHub models as REST endpoints:
  POST /predict/churn      - churn risk prediction
  POST /classify/ticket    - support ticket categorization
  POST /chat                - RAG-powered support chatbot

Run it with:  uv run fastapi dev src/api/main.py
Then open:    http://localhost:8000/docs  (interactive Swagger UI, auto-generated)
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "rag"))

CHURN_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "churn_best_model.joblib"
TICKET_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "ticket_classifier_baseline.joblib"

# Models are loaded ONCE when the server starts, not on every request —
# loading a model from disk on every API call would be extremely slow.
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    models["churn"] = joblib.load(CHURN_MODEL_PATH)
    models["ticket"] = joblib.load(TICKET_MODEL_PATH)

    from src.rag.rag_query import RagPipeline
    models["rag"] = RagPipeline()

    print("All models loaded. API ready.")
    yield
    models.clear()


app = FastAPI(
    title="InsightHub API",
    description="Serving churn prediction, ticket classification, and a RAG support chatbot.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request/response schemas ────────────────────────────────────────
# Pydantic validates every incoming request automatically — if a field
# is missing or the wrong type, FastAPI returns a clear 422 error
# before your code even runs.

class ChurnRequest(BaseModel):
    gender: str
    seniorcitizen: int = Field(ge=0, le=1)
    partner: str
    dependents: str
    tenure: int = Field(ge=0)
    phoneservice: str
    internetservice: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float = Field(ge=0)
    totalcharges: float = Field(ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "seniorcitizen": 0,
                "partner": "Yes",
                "dependents": "No",
                "tenure": 12,
                "phoneservice": "Yes",
                "internetservice": "Fiber optic",
                "contract": "Month-to-month",
                "paperlessbilling": "Yes",
                "paymentmethod": "Electronic check",
                "monthlycharges": 85.5,
                "totalcharges": 1026.0,
            }
        }


class ChurnResponse(BaseModel):
    prediction: str
    churn_probability: float


class TicketRequest(BaseModel):
    subject: str = ""
    description: str

    class Config:
        json_schema_extra = {
            "example": {
                "subject": "Refund not received",
                "description": "I returned my order three weeks ago and still haven't gotten my refund.",
            }
        }


class TicketResponse(BaseModel):
    predicted_category: str
    confidence_scores: dict[str, float]


class ChatRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {"example": {"question": "How long do I have to return an item?"}}


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "InsightHub API is running. Visit /docs for interactive documentation."}


@app.post("/predict/churn", response_model=ChurnResponse)
def predict_churn(request: ChurnRequest):
    try:
        input_data = pd.DataFrame([request.model_dump()])
        # Fill in the columns the model expects that aren't part of this
        # simplified request, using the same defaults as the Streamlit form
        defaults = {
            "multiplelines": "No", "onlinesecurity": "No", "onlinebackup": "No",
            "deviceprotection": "No", "techsupport": "No", "streamingtv": "No",
            "streamingmovies": "No",
        }
        for col, val in defaults.items():
            input_data[col] = val

        model = models["churn"]
        prediction = model.predict(input_data)[0]
        probability = float(model.predict_proba(input_data)[0][1])

        return ChurnResponse(
            prediction="Churn" if prediction == 1 else "No Churn",
            churn_probability=probability,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/ticket", response_model=TicketResponse)
def classify_ticket(request: TicketRequest):
    try:
        combined_text = f"{request.subject} {request.description}".strip()
        model = models["ticket"]

        prediction = model.predict([combined_text])[0]
        probabilities = model.predict_proba([combined_text])[0]
        confidence_scores = {
            category: float(prob) for category, prob in zip(model.classes_, probabilities)
        }

        return TicketResponse(
            predicted_category=prediction,
            confidence_scores=confidence_scores,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        rag = models["rag"]
        result = rag.answer(request.question)
        return ChatResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
