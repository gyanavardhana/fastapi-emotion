from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

# Initialize the sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the custom sentiment scores for each label
sentiment_scores = {
    "sadness": 6,
    "disgust": 5,
    "surprise": 4,
    "fear": 3,
    "anger": 2,
    "neutral": 0,
    "joy": 1
}

# Define the input data model
class Message(BaseModel):
    text: str

@app.post("/analyze-emotion")
async def analyze_emotion(message: Message):
    try:
        # Perform sentiment analysis on the input text
        result = classifier(message.text)[0]
        label = result['label'].lower()  # Get the label in lowercase
        # Look up the score for the detected emotion
        score = sentiment_scores.get(label, 0)  # Default to 0 if label is not found
        return {
            "message": message.text,
            "emotion": label,
            "sentiment_score": score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
