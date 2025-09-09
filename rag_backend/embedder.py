from openai import OpenAI

client = OpenAI(api_key="")

def load_embedding_model():
    return client

def embed_text(text_list, model):
    response = model.embeddings.create(
        input=text_list,
        model="text-embedding-3-small"
    )
    return [embedding.embedding for embedding in response.data]

