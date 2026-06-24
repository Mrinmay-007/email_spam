from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
from .pre_process import transform_text

app = FastAPI(title="Spam Detector API")

# Allow frontend (any origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*",
                   "https://sms-spam-detect.vercel.app"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))


class MessageRequest(BaseModel):
    message: str


class PredictionResponse(BaseModel):
    result: str          # "Spam" or "Not Spam"
    confidence: float    # probability score (0–1)


@app.post("/predict", response_model=PredictionResponse)
async def predict(body: MessageRequest):
    trans_sms = transform_text(body.message)        # Preprocess
    vect_input = tfidf.transform([trans_sms])       # Vectorize
    prediction = model.predict(vect_input)[0]       # Predict label
    proba = model.predict_proba(vect_input)[0]      # Predict probability

    result = "Spam" if prediction == 1 else "Not Spam"
    confidence = float(proba[1])                    # probability of being Spam

    return PredictionResponse(result=result, confidence=confidence)
