from flask import Flask, render_template, request, jsonify
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load FAQ data
with open("data/webdev_faq.json", "r") as f:
    faq = json.load(f)

questions = [item["question"].lower() for item in faq]
answers = [item["answer"] for item in faq]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_response(user_input):
    clean = re.sub(r"[^a-zA-Z ]", "", user_input.lower())
    user_vec = vectorizer.transform([clean])
    similarity = cosine_similarity(user_vec, question_vectors)

    if similarity.max() < 0.3:
        return "I’m not sure about that. Try asking in a different way."

    return answers[similarity.argmax()]

# 1️⃣ Home page
@app.route("/")
def home():
    return render_template("index.html")

# 2️⃣ Chat API
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    bot_reply = get_response(user_message)
    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
