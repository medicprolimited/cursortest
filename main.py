from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://apps.medicpro.london",
        "https://3ad9-184-154-216.ngrok-free.app",  # Correct URL from error message
        "https://3ad9-5-154-184-216.ngrok-free.app", # URL from your HTML file
        "http://localhost:3000",  # For local development if needed
        "http://127.0.0.1:3000",  # For local development if needed
        "http://localhost:5000",  # Common port for local development
        "http://127.0.0.1:5000"   # Common port for local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize HuggingFace client
client = InferenceClient(token=os.getenv("HUGGINGFACE_TOKEN"))

class SimilarityRequest(BaseModel):
    sentence1: str
    sentence2: str

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/api/similarity")
async def compute_similarity(request: SimilarityRequest):
    try:
        # Get embeddings for both sentences
        embedding1 = client.feature_extraction(
            text=request.sentence1,
            model="sentence-transformers/all-mpnet-base-v2"
        )
        embedding2 = client.feature_extraction(
            text=request.sentence2,
            model="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Compute cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)
        
        return {"similarity": float(similarity)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Semantic Similarity API is running"} 