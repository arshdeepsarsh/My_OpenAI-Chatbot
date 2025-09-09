import streamlit as st
from wikipedia_loader import get_wikipedia_summary
from pdf_loader import load_pdf_as_documents
from embedder import load_embedding_model, embed_text
from vector import setup_qdrant, add_to_qdrant, search_qdrant
from generator import generate_answer
import os
import docx

# ---- Page Config ----
st.set_page_config(page_title="IndraAI", page_icon="🤖", layout="wide")

# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "collection_name" not in st.session_state:
    st.session_state.collection_name = "combined_rag"
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

# ---- CSS (UI Styling) ----
st.markdown("""
<style>
body, .stApp {
    background: linear-gradient(135deg, #d5c9ff 0%, #fbc2eb 100%);
    font-family: 'Segoe UI', sans-serif;
    color: #2c2c2c;
}

/* Header */
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 50px;
    background: white;
    border-radius: 50px;
    margin: 10px 40px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.logo { font-size: 22px; font-weight: bold; color: #6A0DAD; }
.menu a {
    margin: 0 15px;
    color: #333;
    text-decoration: none;
    font-weight: 500;
}
.menu a:hover { color: #6A0DAD; }
.start-btn {
    background: #A020F0;
    color: white;
    padding: 8px 22px;
    border-radius: 30px;
    font-weight: 600;
    border: none;
}
.start-btn:hover { background: #6A0DAD; }

/* Hero */
.hero {
    text-align: center;
    margin-top: 40px;
}
.hero h1 {
    font-size: 42px;
    font-weight: bold;
    line-height: 1.3;
}
.hero h1 span {
    color: #6A0DAD;
}
.hero p {
    margin-top: 15px;
    font-size: 18px;
    color: #444;
}

/* Chat Box */
.chat-box {
    background: white;
    border-radius: 16px;
    padding: 12px 16px;
    width: 600px;
    max-width: 92%;
    margin: 30px auto 10px auto;
    display: flex;
    align-items: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    border: 2px solid #6A0DAD;
}
.chat-input {
    flex: 1;
    border: none;
    outline: none;
    font-size: 16px;
    padding: 6px;
}
.chat-send {
    background: #6A0DAD;
    border: none;
    color: white;
    padding: 10px;
    border-radius: 12px;
    cursor: pointer;
}
.chat-send:hover { background: #A020F0; }

/* Upload button */
.plus-btn {
    width:36px; height:36px; border-radius:50%;
    border:1px solid #6A0DAD;
    background:#6A0DAD;
    color:white; font-size:18px; cursor:pointer;
    margin-right:8px;
}
.plus-btn:hover { background:#A020F0; }

/* Disclaimer */
.disclaimer {
    text-align:center;
    color:#444;
    margin-top:12px;
    font-size:13px;
}

/* Chat bubbles */
.user-bubble {
    background:#222; color:white; padding:10px;
    border-radius:8px; margin:10px auto; width:600px; max-width:92%;
}
.ai-bubble {
    background:#f4f4f9; color:black; padding:12px;
    border-radius:8px; margin:10px auto 20px auto;
    width:600px; max-width:92%;
}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div class="header">
  <div class="logo">IndraAI</div>
  <div class="menu">
    <a href="#">Product</a>
    <a href="#">Resources</a>
    <a href="#">Pricing</a>
    <a href="#">Enterprise</a>
  </div>
  <button class="start-btn">Start Building</button>
</div>
""", unsafe_allow_html=True)

# ---- Hero Section ----
st.markdown("""
<div class="hero">
  <h1>Where Curiosity Finds <span>Clarity</span></h1>
  <p>Upload documents, search knowledge, and chat with IndraAI seamlessly.</p>
</div>
""", unsafe_allow_html=True)

# ---- Display Chat ABOVE Search Bar ----
if st.session_state.messages:
    for msg in st.session_state.messages[::-1]:  # latest first
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'><b>You asked:</b> {msg['content']}</div>", unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            st.markdown(f"<div class='ai-bubble'><b>IndraAI:</b> {msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

# ---- Load embedding model ----
model = load_embedding_model()

# ---- Chat Box ----
with st.form("chat_form", clear_on_submit=True):
    c1, c2, c3 = st.columns([0.1, 0.75, 0.15])
    
    with c1:
        plus_clicked = st.form_submit_button("+")
        if plus_clicked:
            st.session_state.show_uploader = not st.session_state.show_uploader

    with c2:
        user_query = st.text_input("", placeholder="Ask your question...", key="chat_text", label_visibility="collapsed")

    with c3:
        submitted = st.form_submit_button("➤")

st.markdown('<div class="disclaimer">IndraAI can make mistakes. Check important info.</div>', unsafe_allow_html=True)

# ---- Show uploader if toggled ----
if st.session_state.show_uploader:
    uploader = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])
else:
    uploader = None

# ---- Handle File Upload ----
if uploader is not None and uploader.name != st.session_state.last_uploaded_name:
    tmp_path = f"tmp_{uploader.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploader.getbuffer())
    
    docs = []
    name = uploader.name.lower()
    try:
        if name.endswith(".pdf"):
            docs = load_pdf_as_documents(tmp_path)
        elif name.endswith(".txt"):
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                docs = [f.read()]
        elif name.endswith(".docx"):
            d = docx.Document(tmp_path)
            docs = [p.text for p in d.paragraphs if p.text.strip()]
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        docs = []
    
    if docs:
        embeddings = embed_text(docs, model)
        if not st.session_state.vector_ready:
            setup_qdrant(st.session_state.collection_name, len(embeddings[0]))
            st.session_state.vector_ready = True
        add_to_qdrant(st.session_state.collection_name, docs, embeddings)
        st.success(f"📄 {uploader.name} added to knowledge base.")
        st.session_state.last_uploaded_name = uploader.name
    
    try:
        os.remove(tmp_path)
    except Exception:
        pass

# ---- Handle Chat Submit ----
if submitted and user_query.strip():
    st.session_state.messages.append({"role": "user", "content": user_query})

    wiki_context = get_wikipedia_summary(user_query)
    context_docs = [wiki_context] if wiki_context else []

    if st.session_state.vector_ready:
        try:
            qvec = embed_text([user_query], model)[0]
            results = search_qdrant(st.session_state.collection_name, qvec, top_k=1)
            if results:
                context_docs.append(results[0])
        except Exception:
            pass

    answer = generate_answer(user_query, context_docs if context_docs else ["No extra context found. Answer briefly."])
    st.session_state.messages.append({"role": "assistant", "content": answer})
