from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from app.nlp import basic_text_features
from app.llm import analyze_deception
from app.speech import transcribe_audio
from app.ml_model import predict_deception
from app.embeddings import compute_similarity
from app.contradiction import contradiction_score

router = APIRouter()

class Statement(BaseModel):
    text:str

@router.post("/analyze")
def analyze(statement: Statement):

    text = statement.text
    features = basic_text_features(text)
    explanation = analyze_deception(text)
    score = predict_deception(text)
    similarity, opposite =contradiction_score(text)

    return{
        "statement": text,
        "features": features,
        "truth_score": score,
        "analysis": explanation,
        "generate_opposite": opposite,
        "contradiction_similarity": similarity
    }

@router.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):

    audio_path = f"temp_{file.filename}"

    with open(audio_path,"wb") as buffer:
        buffer.write(await file.read())
    
    text = transcribe_audio(audio_path)

    explanation = analyze_deception(text)

    return {
        "transcription": text,
        "analysis": explanation
    }