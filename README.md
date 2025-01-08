# Next word prediction using LSTM

This project implements a Next Word Prediction system using LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) neural networks. The system predicts the next word in a sequence based on the input text.

Try the model in action : 


# Key Features
• Text prediction using both LSTM and GRU architectures
• Web interface built with Streamlit for real-time predictions
• Training on Shakespeare's Hamlet text corpus
• Model persistence using H5 and pickle files

# Technical Implementation

**Model Architecture**
• Embedding layer for text vectorization
• Two LSTM/GRU layers with dropout
• Dense output layer with softmax activation
• Early stopping to prevent overfitting

**Data Processing**
• Text tokenization and sequence padding
• Word-to-index mapping
• Input sequence generation
• Train-test split with validation

**Web Application**
• Interactive text input interface
• Real-time predictions
• Model loading and inference capabilities

# Usage
The system accepts text input through a Streamlit web interface and predicts the next most likely word in the sequence. Users can:

• Enter custom text sequences
• Get instant word predictions
• Try different input variations

The project demonstrates practical implementation of sequence prediction using deep learning, with both LSTM and GRU variants available for comparison.
