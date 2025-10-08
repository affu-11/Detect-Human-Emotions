# Detect-Human-Emotions

# Project : # Emotion Detection from Text
# Objective: Detect human emotions (happy, sad, angry, fear, surprise, neutral, etc.) from textual input using a deep learning model.

# Dataset: thttps://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text?resource=download

# Description: Contains text samples (tweets or sentences) labeled with emotions such as joy, anger, sadness, fear, surprise, and love.

# Preprocessing:

Text cleaning: remove punctuation, numbers, stopwords.

Tokenization & padding.

Convert words to vectors using Embedding layer (or pretrained embeddings like GloVe).

Encode labels into one-hot vectors.

# Model Architecture:

Embedding Layer (input_dim = vocab_size, output_dim = 100, input_length = max_len).

Dense(128) – ReLU activation.

Dropout(0.3).

Dense(64) – ReLU activation.

Dense(output_classes) – Softmax activation.

# Training:

Optimizer: Adam.

Loss: Categorical Crossentropy.

Epochs: 15–20.

Batch size: 32 or 64.

# Evaluation:

Accuracy, Precision, Recall, F1-score.

Confusion Matrix to visualize per-class performance.

# Extensions:

Use LSTM/GRU for better sequence modeling.

Build a real-time emotion detection chatbot.

Extend to multilingual emotion detection.

Deploy as a web app (Flask/Streamlit).

# Tools:

TensorFlow/Keras – for ANN modeling.

NLTK/Spacy – for text preprocessing.

scikit-learn – for evaluation metrics.

Matplotlib/Seaborn – for visualization.

# Conclusion:
Detects Human Emotion Based onthe text

