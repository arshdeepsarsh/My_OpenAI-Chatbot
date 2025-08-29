import streamlit as st
from wikipedia_loader import get_wikipedia_summary
from pdf_loader import load_pdf_as_documents
from embedder import load_embedding_model, embed_text
from vector import setup_qdrant, add_to_qdrant, search_qdrant
from generator import generate_answer
import os
import docx

# ---- Page Config ----
st.set_page_config(page_title="My OpenAI-Bot", page_icon="🤖", layout="wide")

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

# ---- Sidebar ----
with st.sidebar:
    st.markdown("<h2 style='color:white;'>📚 OpenAI Chatbot</h2>", unsafe_allow_html=True)
    if st.button("➕ New Chat"):
        if st.session_state.messages:
            st.session_state.history.append(st.session_state.messages.copy())
        st.session_state.messages = []
    st.markdown("### 🔍 Chat History")
    for i, chat in enumerate(st.session_state.history):
        st.markdown(f"**Chat {i+1}:** {len(chat)} messages")

# ---- Main UI ----
st.markdown("<h1 style='text-align:left;'>🤖 My OpenAI Chatbot</h1>", unsafe_allow_html=True)
st.caption("Ask questions. Upload documents. Get AI-powered answers.")

# ---- CSS for Floating Chat Bar ----
st.markdown("""
<style>
.chatbar-wrap {
    position: fixed;
    bottom: 15px;
    left: 50%;
    transform: translateX(-50%);
    width: 100%;
    display: flex;
    justify-content: center;
    z-index: 999;
}
.chatbar {
    width: min(900px, 92vw);
}
.chat-row {
    background:#2c2c2c;
    border-radius:26px;
    padding:6px 10px;
    display:flex;
    align-items:center;
    gap:10px;
}
            
.plus-btn {
    width:36px; height:36px; border-radius:50%;
    border:1px solid #555; background:#2c2c2c;
    color:white; font-size:20px; cursor:pointer;
}
.plus-btn:hover { border-color:#10a37f; color:#10a37f; }
input.chat-input {
    background:transparent; border:none; outline:none;
    color:#fff; width:100%; font-size:16px; padding:8px 6px;
}
button.kbd-send {
    width:40px; height:40px; border-radius:50%;
    border:1px solid #555; background:transparent;
    color:#ddd; cursor:pointer;
}
button.kbd-send:hover { border-color:#10a37f; color:#10a37f; }
.disclaimer { text-align:center; color:#9aa0a6; margin-top:10px; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ---- Display Chat History ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Load embedding model
model = load_embedding_model()

# ---- Floating Chat Bar ----
with st.container():
    st.markdown('<div class="chatbar-wrap"><div class="chatbar">', unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([0.08, 0.84, 0.08])
        
        # Plus Button to toggle uploader
        with c1:
            plus_clicked = st.form_submit_button("+", use_container_width=True)
            if plus_clicked:
                st.session_state.show_uploader = not st.session_state.show_uploader

        with c2:
            user_query = st.text_input("", placeholder="Ask your question...", key="chat_text", label_visibility="collapsed")
        
        with c3:
            submitted = st.form_submit_button("➤", use_container_width=True)
            st.markdown(
                "<script>document.querySelectorAll('button[kind=\"secondary\"]')?.forEach(b=>{b.classList.add('kbd-send')})</script>",
                unsafe_allow_html=True
            )
    
    st.markdown('<div class="disclaimer">My OpenAI-Bot can make mistakes. Check important info.</div>', unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

# Show uploader below chat bar if toggled
if st.session_state.show_uploader:
    uploader = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"])
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

    with st.chat_message("assistant"):
        st.markdown(answer)
