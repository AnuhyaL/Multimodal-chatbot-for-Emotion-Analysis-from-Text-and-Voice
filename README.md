#  Multimodal Chatbot for Emotion Analysis from Text and Voice

A multimodal AI application that analyzes **human emotions from both text and voice input** using deep learning. The project combines a fine-tuned **BERT-based text emotion classifier** with an **LSTM-based speech emotion recognition model**, exposed through a Flask web application.

The system demonstrates how different modalities can be processed independently to identify emotional states from natural human communication.

---

##  Project Overview

Human emotions can be expressed through multiple communication channels. Text can reveal emotional meaning through language, while voice contains additional emotional cues through acoustic characteristics.

This project explores both modalities:

* **Text → BERT → Emotion Classification**
* **Voice → MFCC Features → LSTM → Emotion Classification**
* **Flask → Web Interface / API**

The application accepts either typed text or an uploaded audio file and returns the predicted emotion along with class probabilities.

---

## ✨ Key Features

### 📝 Text Emotion Analysis

* Accepts natural-language text input.
* Uses a fine-tuned **BERT (`bert-base-uncased`)** sequence-classification model.
* Tokenizes and pads text to a maximum sequence length of 128.
* Returns the predicted emotion and probability distribution.
* Supports six emotion categories:

  * Sadness
  * Joy
  * Love
  * Anger
  * Fear
  * Other

### 🎙️ Voice Emotion Analysis

* Accepts uploaded audio files.
* Processes audio using **Librosa**.
* Extracts **13 MFCC features**.
* Standardizes audio to a maximum duration of 5 seconds through truncation/padding.
* Uses a two-layer **LSTM classifier** for emotion recognition.
* Supports seven emotion categories:

  * Angry
  * Disgust
  * Fear
  * Happy
  * Neutral
  * Pleasant Surprise
  * Sad

### 🌐 Flask Web Application

* Flask-based backend.
* REST endpoints for text and audio prediction.
* JSON-based prediction responses.
* Automatically uses CUDA when a compatible GPU is available.
* Includes a browser-based GUI through HTML templates.

---

## 🧠 System Architecture

```text
                    ┌─────────────────────┐
                    │       User          │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │    Flask Web App    │
                    └───────┬───────┬─────┘
                            │       │
                     Text   │       │   Voice
                            │       │
                 ┌──────────▼───┐ ┌─▼──────────────┐
                 │ BERT Tokenizer│ │ Audio Loading  │
                 └──────┬────────┘ └───────┬────────┘
                        │                  │
                 ┌──────▼─────────┐ ┌────▼──────────┐
                 │ Fine-tuned BERT│ │ MFCC Features │
                 │ Classifier     │ └──────┬─────────┘
                 └──────┬─────────┘        │
                        │           ┌──────▼───────┐
                        │           │ LSTM Model    │
                        │           └──────┬────────┘
                        │                  │
                 ┌──────▼──────────────────▼──────┐
                 │        Emotion Prediction       │
                 │     + Probability Scores        │
                 └─────────────────────────────────┘
```

---

## 🛠️ Technologies Used

| Category              | Technologies                    |
| --------------------- | ------------------------------- |
| Programming Language  | Python                          |
| NLP                   | BERT, Hugging Face Transformers |
| Deep Learning         | PyTorch                         |
| Speech Processing     | Librosa                         |
| Speech Features       | MFCC                            |
| Sequence Modeling     | LSTM                            |
| Web Framework         | Flask                           |
| Numerical Computing   | NumPy                           |
| Development           | Jupyter Notebook / Google Colab |
| Hardware Acceleration | CUDA / GPU support              |

---

## 📂 Project Structure

```text
Multimodal-chatbot-for-Emotion-Analysis-from-Text-and-Voice/
│
├── Code/
│   │
│   ├── emotion_text/
│   │   ├── text_emotion_28_04_2025.ipynb
│   │   └── outputs/
│   │
│   └── 30_04_2025_emotion_audio/
│       ├── emotion-detection-30-04-2025.ipynb
│       └── output/
│
├── GUI/
│   ├── app.py
│   └── templates/
│
└── .gitignore
```

The repository separates model development from the application layer, with dedicated notebooks for text and audio emotion recognition and a Flask application for inference.

---

# 🔬 Model Details

## 1. Text Emotion Classification

The text component uses **BERT (`bert-base-uncased`)** with a classification head.

### Pipeline

```text
Input Text
    ↓
BERT Tokenization
    ↓
Padding / Truncation
    ↓
BERT Encoder
    ↓
Classification Head
    ↓
Emotion + Probabilities
```

The training notebook uses a dataset split into:

* Training: 16,000 samples
* Validation: 2,000 samples
* Test: 2,000 samples

The training notebook records a validation accuracy of **93.05% during training**, with the best model saved at that point.

> Note: This README reports the validation result shown in the notebook rather than presenting it as a final independent test-set performance.

### Text Classes

| Class | Emotion |
| ----: | ------- |
|     0 | Sadness |
|     1 | Joy     |
|     2 | Love    |
|     3 | Anger   |
|     4 | Fear    |
|     5 | Other   |

---

## 2. Speech Emotion Classification

The speech component uses an **LSTM neural network** to model temporal patterns in acoustic features.

### Audio Processing Pipeline

```text
Audio Input
    ↓
Resampling
    ↓
5-Second Padding / Truncation
    ↓
MFCC Extraction
    ↓
Temporal Feature Sequence
    ↓
2-Layer LSTM
    ↓
Fully Connected Layer
    ↓
Emotion Prediction
```

The implementation uses:

* Sample rate: `22,050 Hz`
* MFCC features: `13`
* FFT size: `2048`
* Hop length: `512`
* Maximum audio length: `5 seconds`
* LSTM hidden size: `128`
* LSTM layers: `2`
* Dropout: `0.3`

These settings are defined directly in the Flask inference implementation.

### Audio Classes

| Class | Emotion           |
| ----: | ----------------- |
|     0 | Angry             |
|     1 | Disgust           |
|     2 | Fear              |
|     3 | Happy             |
|     4 | Neutral           |
|     5 | Pleasant Surprise |
|     6 | Sad               |

The audio training notebook uses audio organized into these seven emotion categories, including recordings with `OAF` and `YAF` speaker identifiers.

---

# 🚀 Getting Started

## Prerequisites

Make sure you have:

* Python 3.9+
* pip
* Git
* FFmpeg/audio dependencies supported by Librosa

---

## 1. Clone the Repository

```bash
git clone https://github.com/AnuhyaL/Multimodal-chatbot-for-Emotion-Analysis-from-Text-and-Voice.git

cd Multimodal-chatbot-for-Emotion-Analysis-from-Text-and-Voice
```

---

## 2. Create a Virtual Environment

### Windows

```bash
py -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required libraries:

```bash
pip install torch
pip install transformers
pip install flask
pip install numpy
pip install librosa
```

Depending on your operating system and PyTorch configuration, install the appropriate PyTorch build for your CPU/GPU environment.

---

# 📦 Model Files

The Flask application expects trained model weights at:

```text
best_model.pt
full_lstm_model.pth
```

The paths are configured in `GUI/app.py`:

```python
TEXT_MODEL_PATH = 'best_model.pt'
AUDIO_MODEL_PATH = 'full_lstm_model.pth'
```

Place the trained model files in the expected location or update these paths before running the application.

---

# ▶️ Running the Application

Navigate to the GUI directory:

```bash
cd GUI
```

Then start the Flask application:

```bash
python app.py
```

The application runs using Flask's development server.

Open the local application in your browser at:

```text
http://127.0.0.1:5000
```

---

# 🔌 API Endpoints

The Flask backend provides separate endpoints for text and audio emotion analysis.

## Text Prediction

### Endpoint

```text
POST /predict
```

### Request

```json
{
  "text": "I am really happy about the results!"
}
```

### Response

```json
{
  "predicted_class": 1,
  "predicted_emotion": "Joy",
  "probabilities": {
    "Sadness": 0.02,
    "Joy": 0.91,
    "Love": 0.03,
    "Anger": 0.01,
    "Fear": 0.02,
    "Other": 0.01
  }
}
```

---

## Audio Prediction

### Endpoint

```text
POST /predict_audio
```

The endpoint accepts an audio file through a multipart form request.

The backend extracts MFCC features and passes them through the trained LSTM model before returning the predicted emotion and probability distribution.

---

# 📊 Results

### Text Model

| Metric                            |     Result |
| --------------------------------- | ---------: |
| Training Samples                  |     16,000 |
| Validation Samples                |      2,000 |
| Test Samples                      |      2,000 |
| Number of Classes                 |          6 |
| Best Recorded Validation Accuracy | **93.05%** |

The 93.05% figure is the best validation accuracy recorded during the notebook's training process.

### Audio Model

The speech model is trained as a seven-class LSTM classifier using MFCC-based acoustic features.

The project focuses on demonstrating the complete pipeline from:

**raw audio → feature extraction → sequential modeling → emotion classification.**

---

# 🧪 Example Use Cases

This project can serve as a foundation for:

* Emotion-aware conversational AI
* Customer experience analysis
* Human-computer interaction
* Voice-enabled assistants
* Sentiment and emotion monitoring
* Emotion-aware chatbot interfaces
* Multimodal AI research
* Conversational UX applications

---

# ⚠️ Limitations

This project is intended as an **AI/ML research and portfolio project**, not a clinical or psychological assessment system.

Emotion recognition is inherently uncertain and can be affected by:

* Voice quality
* Background noise
* Accent and pronunciation
* Language differences
* Individual speaking styles
* Context and sarcasm
* Dataset bias

The text and audio models also use **different emotion label sets**, so their predictions should not be interpreted as directly equivalent classes.



# 🎯 Learning Outcomes

Through this project, the following concepts were explored:

* Fine-tuning transformer models for NLP classification
* BERT tokenization and sequence classification
* Speech feature extraction using MFCCs
* Sequential modeling with LSTMs
* PyTorch model training and inference
* Audio preprocessing with Librosa
* GPU-accelerated deep learning
* Flask API development
* Integrating trained ML models into a web application
* Multimodal AI system design


# 📚 Project Resources

* **Repository:** [Multimodal Chatbot for Emotion Analysis from Text and Voice](https://github.com/AnuhyaL/Multimodal-chatbot-for-Emotion-Analysis-from-Text-and-Voice)
* **Text Model:** BERT (`bert-base-uncased`)
* **Audio Processing:** Librosa + MFCC
* **Deep Learning:** PyTorch
* **Web Application:** Flask

