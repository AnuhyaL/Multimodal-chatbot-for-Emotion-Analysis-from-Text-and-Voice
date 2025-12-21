import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import io

app = Flask(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Class to label mapping for text emotions
text_class_mapping = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Other"
}

# Class to label mapping for audio emotions
audio_class_mapping = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Pleasant_surprise",
    6: "Sad"
}

# Audio processing configuration
class AudioConfig:
    sample_rate = 22050  # Standard sample rate
    n_mfcc = 13  # Number of MFCC features
    n_fft = 2048  # Size of FFT window
    hop_length = 512  # Hop length for MFCC extraction
    max_audio_length = 5  # Maximum audio length in seconds
    hidden_size = 128
    num_layers = 2
    dropout = 0.3

audio_config = AudioConfig()

# LSTM Model Definition for audio emotion detection
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Get the output from the last time step
        out = self.dropout(out[:, -1, :])
        
        # Pass through the fully connected layer
        out = self.fc(out)
        
        return out

# Load text emotion model
def load_text_model(model_path):
    """Load the trained BERT model with the correct number of labels"""
    try:
        # First try loading the model to inspect its structure
        temp_state_dict = torch.load(model_path, map_location=device)
        # Check the classifier.bias size to determine number of classes
        num_labels = temp_state_dict['classifier.bias'].size(0)
        print(f"Detected {num_labels} classes in the text model")
    except Exception as e:
        print(f"Could not automatically detect number of classes: {e}")
        print("Defaulting to 6 classes based on the error message")
        num_labels = 6

    # Initialize the model with the correct architecture
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )

    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, num_labels

# Load audio emotion model
def load_audio_model(model_path):
    """Load the trained LSTM model for audio emotion detection"""
    try:
        # Check if the model is a full checkpoint or just state dict
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # It's a full checkpoint
            input_size = checkpoint.get('input_size', audio_config.n_mfcc)
            hidden_size = checkpoint.get('hidden_size', audio_config.hidden_size)
            num_layers = checkpoint.get('num_layers', audio_config.num_layers)
            num_classes = checkpoint.get('num_classes', len(audio_class_mapping))
            dropout = checkpoint.get('dropout', audio_config.dropout)
            
            # Initialize model with parameters from checkpoint
            model = LSTMClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout
            )
            
            # Load the state dictionary
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # It's just a model state dict
            model = LSTMClassifier(
                input_size=audio_config.n_mfcc,
                hidden_size=audio_config.hidden_size,
                num_layers=audio_config.num_layers,
                num_classes=len(audio_class_mapping),
                dropout=audio_config.dropout
            )
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        print(f"Audio model loaded successfully with {len(audio_class_mapping)} classes")
        return model
    except Exception as e:
        print(f"Error loading audio model: {e}")
        return None

# Initialize tokenizer for text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set the model paths - update these to your model's location
TEXT_MODEL_PATH = 'best_model.pt'
AUDIO_MODEL_PATH = 'full_lstm_model.pth'  # This would be your audio model path

# Load the models
try:
    text_model, num_labels = load_text_model(TEXT_MODEL_PATH)
    text_model_loaded = True
    print("Text emotion model loaded successfully")
except Exception as e:
    print(f"Could not load text model: {e}")
    print("Text emotion detection will be simulated")
    text_model_loaded = False

# Try to load audio model if it exists
try:
    audio_model = load_audio_model(AUDIO_MODEL_PATH)
    audio_model_loaded = True
    print("Audio emotion model loaded successfully")
except Exception as e:
    print(f"Could not load audio model: {e}")
    print("Audio emotion detection will be simulated")
    audio_model_loaded = False

def predict_text_emotion(text, max_len=128):
    """Predict emotion for a single text input"""
    # If text model is not loaded, return simulated results
    if not text_model_loaded:
        # Simulate prediction with random probabilities
        import random
        
        # Generate random probabilities
        probs = np.random.random(len(text_class_mapping))
        probs = probs / probs.sum()  # Normalize to sum to 1
        
        # Get the class with highest probability
        pred_class = np.argmax(probs)
        
        # Create probability dictionary
        prob_dict = {}
        for i in range(len(probs)):
            class_name = text_class_mapping.get(i, f"Class {i}")
            prob_dict[class_name] = float(probs[i])
        
        return {
            'predicted_class': int(pred_class),
            'predicted_emotion': text_class_mapping.get(pred_class, f"Class {pred_class}"),
            'probabilities': prob_dict
        }
    
    try:
        # Preprocess the text
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move tensors to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Get prediction
        with torch.no_grad():
            outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Get predicted class
        pred_class = torch.argmax(probabilities, dim=1).cpu().numpy()[0]

        # Get class probabilities
        probs = probabilities.cpu().numpy()[0]

        # Create a dictionary of probabilities for all classes
        prob_dict = {}
        for i in range(len(probs)):
            class_name = text_class_mapping.get(i, f"Class {i}")
            prob_dict[class_name] = float(probs[i])

        return {
            'predicted_class': int(pred_class),
            'predicted_emotion': text_class_mapping.get(pred_class, f"Class {pred_class}"),
            'probabilities': prob_dict
        }
    except Exception as e:
        print(f"Error predicting text emotion: {e}")
        # Return a fallback prediction
        return {
            'predicted_class': 1,  # Default to Joy
            'predicted_emotion': "Joy",
            'probabilities': {emotion: 0.1 for emotion in text_class_mapping.values()}
        }

def extract_audio_features(audio_data, sr=None):
    """Extract MFCC features from audio data"""
    try:
        # Load audio from bytes if provided
        if isinstance(audio_data, bytes):
            audio, sr = librosa.load(io.BytesIO(audio_data), sr=audio_config.sample_rate)
        else:
            # Assume audio_data is already a numpy array
            audio = audio_data
            if sr is None:
                sr = audio_config.sample_rate
        
        # Calculate the maximum length to pad/truncate
        max_pad_len = audio_config.sample_rate * audio_config.max_audio_length
        
        # If audio is longer than max_pad_len, truncate it
        if len(audio) > max_pad_len:
            audio = audio[:max_pad_len]
        # If audio is shorter than max_pad_len, pad with zeros
        else:
            audio = np.pad(audio, (0, max_pad_len - len(audio)), 'constant')
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=audio_config.n_mfcc,
            n_fft=audio_config.n_fft, 
            hop_length=audio_config.hop_length
        )
        
        # Transpose to get time as the first dimension (time_steps, n_mfcc)
        mfccs = mfccs.T
        
        return mfccs
    except Exception as e:
        print(f"Error extracting features from audio: {str(e)}")
        return None

def predict_audio_emotion(audio_data):
    """Predict emotion from audio data"""
    # If audio model is not loaded, return simulated results
    if not audio_model_loaded:
        # Simulate prediction with random probabilities
        import random
        
        # Generate random probabilities
        probs = np.random.random(len(audio_class_mapping))
        probs = probs / probs.sum()  # Normalize to sum to 1
        
        # Get the class with highest probability
        pred_class = np.argmax(probs)
        
        # Create probability dictionary
        prob_dict = {}
        for i in range(len(probs)):
            class_name = audio_class_mapping.get(i, f"Class {i}")
            prob_dict[class_name] = float(probs[i])
        
        return {
            'predicted_class': int(pred_class),
            'predicted_label': audio_class_mapping.get(pred_class, f"Class {pred_class}"),
            'probabilities': prob_dict
        }
    
    try:
        # Extract features
        features = extract_audio_features(audio_data)
        
        if features is None:
            raise ValueError("Failed to extract features from audio")
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = audio_model(features_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
        
        # Create probability dictionary
        prob_dict = {}
        for i in range(len(all_probs)):
            class_name = audio_class_mapping.get(i, f"Class {i}")
            prob_dict[class_name] = float(all_probs[i])
        
        return {
            'predicted_class': int(predicted_idx),
            'predicted_label': audio_class_mapping.get(predicted_idx, f"Class {predicted_idx}"),
            'probabilities': prob_dict
        }
    except Exception as e:
        print(f"Error predicting audio emotion: {e}")
        # Return a fallback prediction
        return {
            'predicted_class': 4,  # Default to Neutral
            'predicted_label': "Neutral",
            'probabilities': {emotion: 0.1 for emotion in audio_class_mapping.values()}
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = predict_text_emotion(text)
    return jsonify(result)

@app.route('/predict_audio', methods=['POST'])
def predict_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    # Read audio file
    audio_data = audio_file.read()
    
    # Predict emotion
    result = predict_audio_emotion(audio_data)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
