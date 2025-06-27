from flask import Flask, request, jsonify
from transformers import pipeline

import json

# ✅ Load your local HF pipeline ONCE when server starts
# You can use any small model like flan-t5-small or distilbert for QA or generation
generator = pipeline("text-generation", model="distilgpt2")

app = Flask(__name__)

# ✅ Load a hardcoded Knowledge Base
with open("knowledge_base.json") as f:
    KB = json.load(f)


def match_kb(user_query):
    suggestions = []
    for article in KB["articles"]:
        if any(keyword in user_query.lower() for keyword in article["keywords"]):
            suggestions.append(article)
    return suggestions


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_query = data.get("query")

    # ✅ Use local Hugging Face model
    generated = generator(user_query, max_length=50, num_return_sequences=1)
    hf_answer = generated[0]["generated_text"]

    # ✅ Fallback logic — dummy for now
    fallback = False
    if len(hf_answer) < 20:
        fallback = True

    kb_suggestions = match_kb(user_query)

    return jsonify({
        "answer": hf_answer,
        "kb_suggestions": kb_suggestions,
        "fallback_to_human": fallback
    })


if __name__ == "__main__":
    app.run(debug=True)
