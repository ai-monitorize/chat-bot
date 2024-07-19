import document_embedder as de
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct


client = QdrantClient(url='http://localhost:6333')
# client.create_collection(
#     collection_name="test_collection",
#     vectors_config=models.VectorParams(
#         size=de.get_sentence_embedding_dimension(),
#         distance=models.Distance.COSINE,
#     ),
# )
documents = [
    "Qdrant is a vector similarity search engine.",
    "It allows for efficient similarity searches in large datasets.",
    "It uses advanced indexing techniques."
]

def main():
    points = [
        PointStruct(id=i, vector=de.get_embeddings(documents[i]), payload={"text": documents[i]})
        for i in range(len(documents))
    ]
    client.upsert(
        collection_name='test_collection',
        points=points
    )

main()