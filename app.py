from flask import Flask, request, jsonify, render_template

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import os
import json
import logging

# ✅ Load your Hugging Face text-generation model
generator = pipeline("text-generation", model="distilgpt2")

# ✅ Load a local embeddings model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

app = Flask(__name__)

# ✅ Create logs folder if it doesn't exist
os.makedirs('logs', exist_ok=True)

# ✅ Configure logging
logging.basicConfig(
    filename='logs/queries.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# ✅ Load your KB and precompute embeddings
with open("knowledge_base.json") as f:
    KB = json.load(f)

for article in KB["articles"]:
    # Combine title and keywords for richer embedding
    article["embedding"] = embedder.encode(
        article["title"] + " " + " ".join(article["keywords"])
    )

# ✅ Semantic matcher: computes cosine similarity
def match_kb_semantic(user_query, top_k=2):
    query_embedding = embedder.encode(user_query)
    hits = []

    for article in KB["articles"]:
        sim = util.cos_sim(query_embedding, article["embedding"]).item()
        hits.append((sim, article))

    # Sort by similarity score, descending
    hits.sort(key=lambda x: x[0], reverse=True)

    # Return only JSON-serializable fields
    top_matches = []
    for sim, article in hits[:top_k]:
        clean_article = {
            "id": article["id"],
            "title": article["title"],
            "url": article["url"],
            "keywords": article["keywords"],
            "similarity": round(sim, 4)  # Optional: nice for debugging
        }
        top_matches.append(clean_article)

    return top_matches

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_query = data.get("query")

    # ✅ Generate answer with local LLM
    generated = generator(user_query, max_length=50, num_return_sequences=1)
    hf_answer = generated[0]["generated_text"]

    # ✅ Get semantic KB matches
    kb_suggestions = match_kb_semantic(user_query)

    # ✅ Simple fallback logic
    fallback = False
    if len(hf_answer) < 20:
        fallback = True

    # ✅ Log this request
    logging.info(
        f"Query: {user_query} | "
        f"Answer: {hf_answer.strip()} | "
        f"Fallback: {fallback} | "
        f"KB IDs: {[a['id'] for a in kb_suggestions]}"
    )

    return jsonify({
        "answer": hf_answer.strip(),
        "kb_suggestions": kb_suggestions,
        "fallback_to_human": fallback
    })

# ... your other imports & setup remain the same

@app.route("/")
def home():
    return render_template("index.html")

# ✅ Keep your /predict route as is!


if __name__ == "__main__":
    app.run(debug=True)
