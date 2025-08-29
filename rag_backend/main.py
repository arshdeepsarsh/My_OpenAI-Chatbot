from wikipedia_loader import get_wikipedia_summary
from pdf_loader import load_pdf_as_documents
from embedder import load_embedding_model, embed_text
from vector import setup_qdrant, add_to_qdrant, search_qdrant
from generator import generate_answer
from evaluation import evaluate_response
import tkinter as tk
from tkinter import filedialog

def browse_pdf():
    root = tk.Tk()
    root.withdraw()  # Hides the tkinter main window
    file_path = filedialog.askopenfilename(
        title="Select a PDF file",
        filetypes=[("PDF files", "*.pdf")]
    )
    return file_path


query = input("Enter your question: ")

wiki_context = get_wikipedia_summary(query)
documents = [wiki_context]

pdf_path = input("Enter PDF path (or press Enter to skip): ").strip()
if pdf_path:
    pdf_docs = load_pdf_as_documents(pdf_path)
    documents.extend(pdf_docs)

model = load_embedding_model()
doc_embeddings = embed_text(documents, model)
query_embedding = embed_text([query], model)[0]

COLLECTION_NAME = "combined_rag"
VECTOR_SIZE = len(doc_embeddings[0])
setup_qdrant(COLLECTION_NAME, VECTOR_SIZE)
add_to_qdrant(COLLECTION_NAME, documents, doc_embeddings)

top_doc = search_qdrant(COLLECTION_NAME, query_embedding, top_k=1)[0]
short_doc = top_doc[:150].replace("\n", " ").strip()
print("\n Top Retrieved Document (preview):\n", short_doc + "...")


answer = generate_answer(query, [top_doc])
print("\n Final Answer:\n", answer)

