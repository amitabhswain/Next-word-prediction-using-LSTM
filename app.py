import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
try:
    model = load_model('next_word_lstm.keras')
except Exception as e:
    st.error("Failed to load the LSTM model. Please check the file path and ensure the model file exists.")
    raise e

# Load the tokenizer
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error("Failed to load the tokenizer. Please check the file path and ensure the tokenizer file exists.")
    raise e

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    """Predict the next word based on the input text."""
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]  # Convert text to sequence of integers
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len - 1):]  # Truncate to match max_sequence_len-1
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')  # Pad sequence
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]  # Extract the predicted word index

        # Map the index back to the corresponding word
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
    return None

# Streamlit app
st.title("Next Word Prediction with LSTM")
st.write("Enter a sequence of words to predict the next word.")

# User input
input_text = st.text_input("Enter the sequence of words", "To be or not to")

# Predict button
if st.button("Predict Next Word"):
    if input_text.strip():  # Check if input is not empty
        try:
            max_sequence_len = model.input_shape[1] + 1  # Retrieve max sequence length from model input shape
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
            if next_word:
                st.write(f"Next word: **{next_word}**")
            else:
                st.write("Could not predict the next word. Try a different input sequence.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
    else:
        st.write("Please enter a valid sequence of words.")
