import re
import json
from flask import Flask, request, render_template_string, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "super_secret_key"

# ---------- NORMALIZATION ----------
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- LOAD DATA ----------
with open("data/webdev_faq.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions = [normalize(item["question"]) for item in faq_data]
answers = [item["answer"] for item in faq_data]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# ---------- CHAT LOGIC ----------
def get_best_answer(user_input):
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)

    if similarity.max() < 0.2:
        return "I'm not sure about that. Try asking in a different way."

    return answers[similarity.argmax()]

# ---------- UI ----------
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>Web Dev Assistant</title>
    <style>
        body { font-family: Arial; background:#f4f4f4; display:flex; justify-content:center; padding:20px; }
        .box { width:400px; background:white; border-radius:8px; overflow:hidden; }
        .header { background:#007bff; color:white; padding:15px; text-align:center; }
        .chat { height:350px; overflow-y:auto; padding:15px; }
        .user {
    text-align: right;
    margin: 10px 0;
    background: #d1ecf1;
    padding: 10px;
    border-radius: 10px;
    max-width: 75%;
    margin-left: auto;
}

.bot {
    text-align: left;
    margin: 10px 0;
    background: #f1f1f1;
    padding: 10px;
    border-radius: 10px;
    max-width: 75%;
    margin-right: auto;
}

        .input { display:flex; border-top:1px solid #ddd; }
        input { flex:1; padding:10px; border:none; outline:none; }
        button { padding:10px; background:#007bff; color:white; border:none; }
    </style>
</head>
<body>
<div class="box">
    <div class="header">Asrar's Web Dev Assistant</div>
    <div class="chat">
        {% for h in history %}
            <div class="user"><b>You:</b> {{ h.user }}</div>
            <div class="bot"><b>Bot:</b> {{ h.bot }}</div>
        {% endfor %}
    </div>
    <form method="post" class="input">
        <input name="user_input" placeholder="Ask something..." required>
        <button>Send</button>
    </form>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def chat():
    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        raw = request.form["user_input"]
        cleaned = normalize(raw)
        reply = get_best_answer(cleaned)

        session["history"].append({"user": raw, "bot": reply})
        session.modified = True

    return render_template_string(HTML_TEMPLATE, history=session["history"])

if __name__ == "__main__":
    app.run(debug=True)
