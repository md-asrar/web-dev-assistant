# Web Development Assistant Chatbot

This project is a simple rule-based Web Development Assistant chatbot built using Python and Flask.  
It answers beginner-level questions related to HTML, CSS, JavaScript, Flask, Git, and general web development concepts.

The chatbot works by matching user questions with predefined questions stored in a JSON file and returning the most relevant answer using basic NLP techniques.

---

## Features

- Web-based chatbot interface
- Answers common web development questions
- Supports greetings and general queries
- Uses TF-IDF and cosine similarity for matching
- Built with Python and Flask
- Beginner-friendly and easy to extend

---

## Technologies Used

- Python
- Flask
- HTML & CSS
- JSON (for chatbot knowledge base)
- scikit-learn (TF-IDF & cosine similarity)

---

## Project Structure

web-dev-assistant/
│
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── data/
    └── webdev_faq.json     # Questions and answers used by the chatbot

---

## How to Run the Project Locally

1. Open terminal / command prompt  
2. Navigate to the project folder  
3. Install dependencies:
   
   python -m pip install -r requirements.txt

4. Run the application:

   python app.py

5. Open your browser and go to:

   http://127.0.0.1:5000/

---

## How to Add More Questions

- Open `data/webdev_faq.json`
- Add new question–answer pairs in the same format
- Restart the Flask server

---

## Author

Md Asrar

---

## Note

This project is created for learning purposes and academic/certification submission.  
It is intentionally simple to demonstrate understanding of Flask, basic NLP, and project structure.
