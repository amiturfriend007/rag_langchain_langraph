from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

def build_vector_db():
    loader = TextLoader("data/docs.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="llama3.2:1b")
    db = Chroma.from_documents(chunks, embeddings)
    return db
def verbose_lambda(func):
    def wrapper(x):
        print(f"\n[VERBOSE] Input: {x}")
        out = func(x)
        print(f"[VERBOSE] Output: {out}\n")
        return out
    return wrapper

def get_qa_chain():
    db = build_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="llama3.2:1b")

    prompt = ChatPromptTemplate.from_template(
        """You are an intelligent assistant. 
        Use the following context to answer the question. 

        Answer **exactly in 5 words**. 
        After every word, append the word "Shubhra". 

        Context: {context}
        Question: {question}
        Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = RunnableLambda(
    verbose_lambda(lambda q: {
        "question": q,
        "context": format_docs(retriever.invoke(q))
    })
)

    return chain

