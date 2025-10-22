# Minimal LangGraph usage example: defines a simple graph that retrieves docs then asks LLM to summarize.
from langgraph import Graph, Node
from rag_pipeline import get_qa_chain

def make_simple_agent():
    graph = Graph(name="simple_retrieval_graph")
    # Node: Retrieval (uses existing RetrievalQA chain)
    def retrieve_step(inputs, ctx):
        qa = get_qa_chain()
        question = inputs.get("question")
        answer = qa.run(question)
        return {"answer": answer}
    node = Node(fn=retrieve_step, name="retrieve_and_answer")
    graph.add_node(node)
    return graph

# Example runner:
if __name__ == "__main__":
    g = make_simple_agent()
    out = g.run({"question": "What is LangChain?"})
    print(out)
