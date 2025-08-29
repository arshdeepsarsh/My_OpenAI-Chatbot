from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

client = QdrantClient("localhost", port=6334)

def setup_qdrant(collection_name, vector_size):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

def add_to_qdrant(collection_name, documents, embeddings):
    points = [
        PointStruct(id=uuid.uuid4().int % (10 ** 8), vector=emb, payload={"text": doc})
        for doc, emb in zip(documents, embeddings)
    ]
    client.upsert(collection_name=collection_name, points=points)

def search_qdrant(collection_name, query_vector, top_k=1):
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    return [hit.payload["text"] for hit in hits]

