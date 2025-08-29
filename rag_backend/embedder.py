import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def load_embedding_model():
    return client

def embed_text(text_list, model):
    response = model.embeddings.create(
        input=text_list,
        model="text-embedding-3-small"
    )
    return [embedding.embedding for embedding in response.data]
