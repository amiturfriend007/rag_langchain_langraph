from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import get_qa_chain, build_vector_db
from graph_agent import make_simple_agent
import os

app = FastAPI(title="RAG Full AI Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy initialize - call build_vector_db() first time if chroma db missing
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "db", "chroma_db")
if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    try:
        build_vector_db()
    except Exception as e:
        print("Warning: build_vector_db failed on startup:", e)

qa_chain = get_qa_chain()

@app.post("/api/query")
async def query_rag(question: str = Form(...)):
    try:
        res = qa_chain.run(question)
    except Exception as e:
        res = f"Error running QA chain: {e}"
    return {"answer": res}

@app.post("/api/agent")
async def run_agent(question: str = Form(...)):
    try:
        g = make_simple_agent()
        out = g.run({"question": question})
    except Exception as e:
        out = {"error": str(e)}
    return out

@app.post("/api/embed")
async def rebuild_embeddings():
    try:
        build_vector_db()
        return {"status": "ok", "message": "Embeddings rebuilt"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
