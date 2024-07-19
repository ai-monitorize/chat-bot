from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
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
        self.collection_name = 'test_collection'
        self.vector_size = self.model.get_sentence_embedding_dimension()

    def init_collection(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            )
        )

    def upsert(self, embeddings: [VectorEmbedding]):
        points = [
            PointStruct(id=embeddings[i].id, vector=self.model.encode(embeddings[i].chunk).tolist(),
                        payload=embeddings[i].payload)
            for i in range(len(embeddings))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 1, query_filter: Filter = None):
        query_vector = self.model.encode(query)
        return self.client.search(collection_name=self.collection_name, query_vector=query_vector, limit=top_k,
                                  query_filter=query_filter)
