from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize the sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")

# Initialize the Flask app
app = Flask(__name__)

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

@app.route('/analyze-emotion', methods=['POST'])
def analyze_emotion():
    try:
        # Parse the JSON request data
        data = request.get_json()
        text = data.get('text', "")

        # Perform sentiment analysis on the input text
        result = classifier(text)[0]
        label = result['label'].lower()  # Get the label in lowercase
        # Look up the score for the detected emotion
        score = sentiment_scores.get(label, 0)  # Default to 0 if label is not found

        # Create the response
        response = {
            "message": text,
            "emotion": label,
            "sentiment_score": score
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Set debug=True for development purposes
