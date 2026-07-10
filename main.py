from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import sqlite3
from datetime import datetime

app = FastAPI()

# Allow requests from any origin (fine for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# ---------- Database setup ----------
def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            checked_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()  # Runs once when the server starts


# ---------- Request/response models ----------
class MessageInput(BaseModel):
    text: str


# ---------- Routes ----------
@app.get("/")
def read_root():
    return {"message": "Spam Classifier API is running!"}


@app.post("/predict")
def predict(input: MessageInput):
    text_tfidf = vectorizer.transform([input.text])
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]

    label = "spam" if prediction == 1 else "ham"
    confidence = float(probability[prediction] * 100)
    timestamp = datetime.now().isoformat()

    # Save this check into the database
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO history (message, prediction, confidence, checked_at) VALUES (?, ?, ?, ?)",
        (input.text, label, round(confidence, 2), timestamp)
    )
    conn.commit()
    conn.close()

    return {
        "message": input.text,
        "prediction": label,
        "confidence": round(confidence, 2)
    }


@app.get("/history")
def get_history(limit: int = 20):
    conn = sqlite3.connect("history.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


@app.delete("/history")
def clear_history():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return {"message": "History cleared"}