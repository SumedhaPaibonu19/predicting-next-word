import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained model and tokenizer
model = load_model('hamlet_lstm_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

#function to predict the next word
def predict_next_word(mode, tokenizer, text, max_sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_length:
        token_list = token_list[-(max_sequence_length-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

#streamlit app
st.title("Hamlet Next Word Predictor")
input_text = st.text_input("Enter the sequence of words")
if st.button("Predict Next Word"):
    max_sequence_length = model.input_shape[1] +1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
    if next_word:
        st.write(f"The next word is: {next_word}")
    else:
        st.write("Could not predict the next word.")