from fastapi import FastAPI, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rag_backend.wikipedia_loader import get_wikipedia_summary
from rag_backend.pdf_loader import load_pdf_as_documents
from rag_backend.embedder import load_embedding_model, embed_text
from rag_backend.vector import setup_qdrant, add_to_qdrant, search_qdrant
from rag_backend.generator import generate_answer
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello, your API is running!"}

@app.post("/ask")
async def ask_question(query: str = Form(...), pdf: UploadFile = None):
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    wiki_context = get_wikipedia_summary(query)
    documents = [wiki_context] if wiki_context else []

    if pdf:
        tmp = f"temp_{pdf.filename}"
        with open(tmp, "wb") as f:
            f.write(await pdf.read())
        docs = load_pdf_as_documents(tmp)
        documents.extend(docs)
        os.re
