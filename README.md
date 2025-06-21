This project is a Hamlet-themed Next Word Predictor, implemented as a Streamlit web application. It leverages a deep learning model (LSTM) trained on the full text of Shakespeare's Hamlet to predict the most probable next word based on a given input sequence. 

Features
Interactive Web Application: A user-friendly interface built with Streamlit allows for easy interaction. 
Next Word Prediction: Predicts the next word in a sequence based on a trained LSTM model. 
Shakespearean Context: The model is specifically trained on "shakespeare-hamlet.txt" from the NLTK Gutenberg corpus, providing predictions relevant to the play's vocabulary and style.
Technologies Used
The project uses the following key technologies:
```
Python 
Streamlit 
TensorFlow (specifically Keras) 
Numpy 
Pandas 
NLTK 
Scikit-learn 
TensorBoard 
Matplotlib 
Scikeras
```
Setup and Installation
To set up and run this project locally, follow these steps:
```
git clone https://github.com/your-username/predicting-next-project.git
cd predicting-next-project
Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
pip install -r requirements.txt
```
