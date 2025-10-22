RAG Full AI Demo
=================

Structure:
- backend/: FastAPI + LangChain + LangGraph + Chroma + Llama integration
- frontend/: Next.js simple UI (calls backend endpoints)

Local LLM:
- The project is configured to use `llama3.2:1b` via the Ollama wrapper (Ollama client in LangChain).
- Ensure your local LLM server or wrapper supports that model name, or update the model name in backend/rag_pipeline.py

Quick start (backend):
1. python -m venv .venv
2. source .venv/bin/activate    # on Windows use .venv\Scripts\activate
3. pip install -r backend/requirements.txt
4. cd backend
5. uvicorn main:app --reload --port 8000

Quick start (frontend):
1. cd frontend
2. npm install
3. npm run dev
4. Open http://localhost:3000

Notes:
- This is a starter template. You may need to adapt embedding/LLM classes depending on your local Ollama/embedding provider.
- LangGraph usage here is minimal — replace with more advanced graphs/workflows as needed.
