from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class vectorDataBase:
    def __init__(self):
        pass
    def vectorBD(self, vectors_collection, text_tokens):
        try:
            client = QdrantClient(url="http://localhost:6333")
            client.create_collection(
                    collection_name="test_collection",
                    vectors_config=VectorParams(size=768, distance=Distance.DOT),
            )
            from qdrant_client.models import PointStruct

            operation_info = client.upsert(
                    collection_name="test_collection",
                    wait=True,
                    points=[
                        PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
                        PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
                        PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
                        PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
                        PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
                        PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),

                        # PointStruct(id, vector, payload for id, vector, payload in zip(enumerate(vectors_collection), text_tokens))
                    ],
            )

            print(operation_info)
        except vectorDataBase as vdbe:
            raise("The DB operation Failed..", vdbe.__getstate__)
