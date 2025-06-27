# AI Customer Support Assistant

A GPT-powered AI assistant that handles customer queries, suggests knowledge base (KB) articles, and escalates complex tickets to human agents.

##  Features
- OpenAI GPT integration for understanding and generating responses
- Hardcoded KB suggestions using keyword matching
- Fallback to human agent if confidence is low
- Flask REST API backend

##  Tech Stack
- Python
- Flask
- OpenAI API
- JSON for KB store

## 📂 Folder Structure
.
├── app.py
├── knowledge_base.json
├── .env
├── requirements.txt
├── README.md

markdown
Copy
Edit

## ⚙️ Running Locally
1. Clone the repo and create a virtual environment.
2. Add your OpenAI API key to `.env`:
OPENAI_API_KEY=your_key_here

3. Install requirements:
pip install -r requirements.txt


4. Run:
python app.py


5. Test using Postman or `curl`.

## 📌 Next Steps
- Add caching with Redis
- Improve KB matching with embeddings
- Deploy to Heroku
- Add monitoring & logging
