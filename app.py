from flask import Flask, render_template, request, jsonify
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------- LOAD DATA ----------------
with open("data/webdev_faq.json", "r", encoding="utf-8") as f:
    faq = json.load(f)

questions = [item["question"].lower() for item in faq]
answers = [item["answer"] for item in faq]

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)
question_vectors = vectorizer.fit_transform(questions)

# ---------------- CORE LOGIC ----------------
def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

def get_response(user_input):
    clean = normalize(user_input)

    # --------- STRONG INTENT HANDLING ---------
    if any(word in clean for word in ["hi", "hello", "hey", "good morning", "good evening"]):
        return "Hello! I’m your Web Development Assistant. You can ask me about HTML, CSS, JavaScript, Flask, Git, and web development concepts."

    if "javascript" in clean or clean == "js":
        return (
            "JavaScript is a programming language used to make websites interactive. "
            "It handles things like button clicks, form validation, dynamic content, APIs, and animations."
        )

    if "html" in clean or "htlm" in clean:
        return (
            "HTML (HyperText Markup Language) is used to structure web pages. "
            "It defines elements like headings, paragraphs, images, links, and forms."
        )

    if "css" in clean:
        return (
            "CSS (Cascading Style Sheets) is used to style web pages. "
            "It controls layout, colors, fonts, responsiveness, and overall design."
        )

    if "what can you do" in clean or "your abilities" in clean or "what do you do" in clean:
        return (
            "I can explain web development concepts, answer questions about HTML, CSS, JavaScript, Flask, Git, "
            "and guide beginners in building web applications."
        )

    # --------- NLP SIMILARITY FALLBACK ---------
    user_vec = vectorizer.transform([clean])
    similarity = cosine_similarity(user_vec, question_vectors)

    best_score = similarity.max()
    best_index = similarity.argmax()

    if best_score < 0.35:
        return (
            "I’m not fully sure about that yet. "
            "Try asking in a simpler way or ask about HTML, CSS, JavaScript, Flask, or Git."
        )

    return answers[best_index]

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    reply = get_response(user_message)
    return jsonify({"reply": reply})

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
