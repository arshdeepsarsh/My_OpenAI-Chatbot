import pdfplumber

def load_pdf_as_documents(file_path):
    documents = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    documents.append(text.strip())
    except Exception as e:
        print(f"Error loading PDF: {e}")
    return documents
