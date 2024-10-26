from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Load the emotion classifier model
pipe_lr = joblib.load(open("emotion_classifier_pipe_lr_16_june_2021.pkl", "rb"))

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

# Prediction functions for the emotion classifier
def predict_emotion(text: str):
    result = pipe_lr.predict([text])
    return result[0]

def get_prediction_proba(text: str):
    result = pipe_lr.predict_proba([text])
    return result

@app.post("/predict-emotion")
async def predict_emotion_api(message: Message):
    try:
        # Perform prediction and probability calculation
        prediction = predict_emotion(message.text).lower()  # Get label in lowercase
        probability = get_prediction_proba(message.text)
        
        # Define the emoji dictionary
        emotions_emoji_dict = {
            "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
            "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
            "shame": "😳", "surprise": "😮"
        }
        
        # Look up the score and emoji
        score = sentiment_scores.get(prediction, 0)  # Default to 0 if not found
        emoji_icon = emotions_emoji_dict.get(prediction, "")
        confidence = np.max(probability) * 100

        return {
            "message": message.text,
            "emotion": prediction,
            "emoji": emoji_icon,
            "sentiment_score": score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hello")
async def hello_world():
    return "hello world"
