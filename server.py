import os
import io
import torch
import soundfile as sf
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import cohere
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained emotion detection model
MODEL_PATH = "my_trained_model.pth"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=7)

# Load model state dictionary
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
else:
    raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' not found!")

model.eval()  # Set model to evaluation mode

# Emotion label mapping
LABEL_MAP = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "fearful", 5: "disgust", 6: "surprised"}

@app.route("/predict", methods=["POST"])
def predict_emotion():
    """Handles audio file input and returns predicted emotion."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        audio_data, samplerate = sf.read(io.BytesIO(file.read()), dtype="float32")
    except Exception as e:
        return jsonify({"error": f"Failed to read audio file: {e}"}), 400

    # Preprocess audio for model
    inputs = processor(audio_data, sampling_rate=samplerate, return_tensors="pt", padding=True, return_attention_mask=False)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = torch.argmax(logits, dim=-1).item()
    emotion = LABEL_MAP.get(predicted_label, "unknown")

    return jsonify({"emotion": emotion})

# Initialize Cohere API
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("❌ Cohere API key missing. Check .env file.")

co = cohere.Client(COHERE_API_KEY)

@app.route("/cohere_response", methods=["POST"])
def get_cohere_response():
    """Receives text and generates a response using Cohere."""
    data = request.json
    if not data or "text" not in data or "emotion" not in data:
        return jsonify({"error": "Missing required fields: 'text' and 'emotion'"}), 400

    user_text = data["text"].strip()
    user_emotion = data["emotion"].strip()

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    try:
        response = co.chat(
            model="command-r-plus",
            message=f"User is feeling {user_emotion}. They said: {user_text}"
        )
        return jsonify({"response": response.texts[0].strip()})
    except Exception as e:
        return jsonify({"error": f"Cohere API Error: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)


