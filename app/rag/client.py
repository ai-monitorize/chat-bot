from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from sentence_transformers import SentenceTransformer


class VectorEmbedding:
    def __init__(self, vector_id, payload, chunk):
        self.id = vector_id
        self.payload = payload
        self.chunk = chunk


class VectorEngineClient:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.client = QdrantClient(url='http://localhost:6333')
        self.collection_name = 'bla'

    def search(self, query: str, top_k: int = 1, query_filter: Filter = None):
        query_vector = self.model.encode(query)
        return self.client.search(collection_name=self.collection_name, query_vector=query_vector, limit=top_k,
                                  query_filter=query_filter)
