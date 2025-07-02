import gradio as gr
import pickle
from nltk.tokenize import word_tokenize
import nltk

# Load models
model = pickle.load(open('model.pkl', 'rb'))         # Doc2Vec model
xgb_best = pickle.load(open('xgb_best.pkl', 'rb'))   # XGBoost model

# Download tokenizer
nltk.download('punkt')

# Prediction function
def classify_tweet(text):
    try:
        tokens = word_tokenize(text.lower())
        vector = model.infer_vector(tokens).reshape(1, -1)
        prediction = xgb_best.predict(vector)[0]
        return "Disaster" if prediction == 1 else "Not a Disaster"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=classify_tweet,
    inputs=gr.Textbox(lines=10, placeholder="Enter a tweet..."),
    outputs="text",
    title="Disaster Tweet Classifier",
    description="Enter a tweet to check if it's related to a disaster or not."
)

interface.launch()
