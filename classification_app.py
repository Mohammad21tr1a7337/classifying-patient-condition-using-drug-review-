import streamlit as st
import pickle
import html
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Set Streamlit page config FIRST
st.set_page_config(page_title="Medical Condition Predictor", layout="centered")

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Background and style
st.markdown("""
<style>
.stApp {
    background-image: url('https://img.freepik.com/free-vector/medical-seamless-pattern_1284-41627.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: black !important;
}
.box {
    background-color: rgba(255,255,255,0.85);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
textarea {
    background-color: rgba(255,255,255,0.9);
    color: black;
    font-size: 16px;
}
.stButton>button {
    background-color: #00897B;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    padding: 0.5rem 1rem;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# Preprocessing function
def preprocess_review(text):
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'https?://\\S+|www\\.\\S+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r"\d+", "", text)
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Label mapping
condition_map = {
    0: "Depression",
    1: "Diabetes Type 2",
    2: "High Blood Pressure"
}

# UI
st.markdown("<div class='box'><h1>🩺 Medical Condition Predictor</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='box'><p>Enter patient review or symptom description below:</p></div>", unsafe_allow_html=True)
user_input = st.text_area("Patient Review", height=150)

if st.button("🔍 Predict Condition"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a valid review.")
    else:
        cleaned = preprocess_review(user_input)
        vec_input = vectorizer.transform([cleaned])
        pred = model.predict(vec_input)[0]
        label = condition_map.get(pred, "Unknown")

        # Color scheme
        if label == "Depression":
            bg_color = "#f3e5f5"
            emoji = "💭"
            font_color = "#6A1B9A"
        elif label == "Diabetes Type 2":
            bg_color = "#e1f5fe"
            emoji = "🍬"
            font_color = "#0277BD"
        else:
            bg_color = "#ffebee"
            emoji = "❤️"
            font_color = "#C62828"

        st.markdown(f"""
        <div style='background-color:{bg_color};padding:20px;border-radius:10px;margin-top:20px'>
            <h3 style='color:{font_color};'>{emoji} Predicted Condition: <b>{label}</b></h3>
        </div>
        """, unsafe_allow_html=True)
