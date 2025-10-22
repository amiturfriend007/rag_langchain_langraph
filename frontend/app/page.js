"use client";
import { useState } from "react";

export default function Page() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [mode, setMode] = useState("rag");

  const ask = async () => {
    const form = new FormData();
    form.append(mode === "rag" ? "question" : "question", question);
    const url = mode === "rag" ? "http://localhost:8000/api/query" : "http://localhost:8000/api/agent";
    const res = await fetch(url, { method: "POST", body: form });
    const data = await res.json();
    setAnswer(JSON.stringify(data, null, 2));
  };

  return (
    <main style={{ padding: 24 }}>
      <h1>RAG Full AI Demo (llama3.2:1b)</h1>
      <div style={{ marginTop: 12 }}>
        <label>
          Mode:{" "}
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="rag">RAG</option>
            <option value="agent">Agent (LangGraph)</option>
          </select>
        </label>
      </div>
      <textarea
        placeholder="Ask something..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        rows={6}
        style={{ width: "100%", marginTop: 12 }}
      />
      <div style={{ marginTop: 12 }}>
        <button onClick={ask}>Ask</button>
      </div>
      <pre style={{ marginTop: 12, background: "#f2f2f2", padding: 12 }}>{answer}</pre>
    </main>
  );
}
