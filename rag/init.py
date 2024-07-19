from client import VectorEngineClient, VectorEmbedding

client = VectorEngineClient()
documents = [
    "As for vector similarity search engine EON uses Qdrant.",
    "It allows for efficient similarity searches in large datasets.",
    "It uses advanced indexing techniques."
]


def main():
    client.init_collection()
    points = [
        VectorEmbedding(vector_id=i, chunk=documents[i], payload={"text": documents[i]})
        for i in range(len(documents))
    ]
    client.upsert(points)


main()
