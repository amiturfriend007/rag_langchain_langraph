from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from rag_pipeline import get_qa_chain

# -------------------------------
# FastAPI
# -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Models
# -------------------------------
llm = OllamaLLM(model="llama3.2:1b")

# Example “tool”
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

rag_chain = get_qa_chain()

# Wrap the tool as RunnableLambda
tools_runnable = RunnableLambda(lambda q: calculator(q))

# Wrap LLM
agent_runnable = RunnableLambda(lambda q: llm.invoke(q))

# Combine: first call tool, then LLM
agent_chain = RunnableMap({
    "tool_output": tools_runnable,
    "llm_output": agent_runnable
})

# -------------------------------
# API
# -------------------------------
class Query(BaseModel):
    question: str


import logging
logging.basicConfig(level=logging.INFO)

def agent_logic(question: str) -> str:
    if any(c in question for c in "+-*/"):
        reasoning = f"Detected math expression in: '{question}'. Using calculator."
        logging.info(reasoning)
        result = calculator(question)
    elif "document" in question.lower() or "context" in question.lower():
        reasoning = f"Detected context-based query in: '{question}'. Using RAG pipeline."
        logging.info(reasoning)
        result = rag_chain.invoke(question)
    else:
        reasoning = f"General query detected: '{question}'. Using LLM."
        logging.info(reasoning)
        result = llm.invoke(question)
    
    logging.info(f"Final result: {result}")
    return result

def format_response(response):
    words = response.split()[:5]  # Take first 5 words
    return " ".join([f"{word}Shubhra" for word in words])

@app.post("/api/agent")
async def agent_query(query: Query):
    try:
        response = agent_logic(query.question)
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/")
async def root():
    return {"message": "Agent backend running"}
