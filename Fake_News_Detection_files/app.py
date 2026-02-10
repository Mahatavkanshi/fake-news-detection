
import gradio as gr
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and TF-IDF vectorizer
model = pickle.load(open("models/logistic_model.pkl", "rb"))
tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]
    return " ".join(cleaned_words)

def predict_news(news_text):
    cleaned = clean_text(news_text)
    vector = tfidf.transform([cleaned])
    result = model.predict(vector)[0]
    return "Real News" if result == 1 else "Fake News"

interface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=6, placeholder="Paste news article here..."),
    outputs="text",
    title="Fake News Detection System",
    description="Enter a news article to check whether it is Fake or Real."
)

if __name__ == "__main__":
    interface.launch()
