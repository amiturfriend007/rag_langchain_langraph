from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings  # placeholder: depending on your local setup you may swap
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import os

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "db", "chroma_db")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "documents")

def build_vector_db(persist=True):
    # Load text files
    loaders = []
    for fname in os.listdir(DATA_PATH):
        if fname.endswith(".txt"):
            loaders.append(TextLoader(os.path.join(DATA_PATH, fname), encoding="utf-8"))
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # NOTE: OllamaEmbeddings is used as a placeholder; configure according to your embeddings provider.
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    if persist:
        db.persist()
    return db

def get_qa_chain():
    # Ensure DB exists (you can call build_vector_db() to (re)create)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = db.as_retriever()
    # Use local Llama model via Ollama wrapper; model name set to llama3.2:1b as requested
    llm = Ollama(model="llama3.2:1b", temperature=0.0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa_chain
