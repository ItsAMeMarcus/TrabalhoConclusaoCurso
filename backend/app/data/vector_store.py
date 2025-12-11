# app/services/vector_store.py
from typing import Any, Dict, List
import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance

class TccVectorStore:
    def __init__(self, collection_name, distancia, vector_size=1024):
        # NOTA: O BERT (neuralmind/bert-base-portuguese-cased) usa 768 dimensões, não 384.
        # Se usar outro modelo, ajuste o vector_size.
        
        # Conecta ao Qdrant (usando o nome do serviço no docker-compose ou localhost)
        self.client = QdrantClient(url="http://localhost:6333") 
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distancia
        
        # Garante que a coleção existe ao iniciar
        if not self.client.collection_exists(collection_name):
            self.setup_collection()
        else:
            print(f"Conectado à coleção existente: {self.collection_name}")

    def setup_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size, 
                distance=self.distance, 
                on_disk=True
            ),

            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, 
                    quantile=0.99, 
                    always_ram=True
                )
            ),
            on_disk_payload=True
        )
        # Índices de metadados
        for field in ["autor", "orientador", "titulo", "ano", "endereco"]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD
            )

    def upsert_chunks(self, chunks_data: List[Dict[str, Any]]):
        """
        Recebe uma lista de dicionários contendo: 'text', 'embedding', 'metadata'
        """
        points = []
        for item in chunks_data:
            # Validação de segurança para evitar erro de dimensão
            if len(item["embedding"]) != self.vector_size:
                raise ValueError(f"ERRO CRÍTICO: O modelo gerou vetor de tamanho {len(item['embedding'])}, "
                                 f"mas a coleção espera {self.vector_size}. Verifique seu modelo.")

            # ID Determinístico (Evita duplicatas se o agente tentar de novo)
            texto_content = item["text"]
            id_unico = str(uuid.uuid5(uuid.NAMESPACE_DNS, texto_content))

            points.append(
                models.PointStruct(
                    id=id_unico,
                    vector=item["embedding"],
                    payload={
                        "texto_chunk": item["text"],
                        **item["metadata"] # Espalha autor, ano, titulo, ordem
                    }
                )
            )
        
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        
        if operation_info.status != "completed":
            raise Exception(f"Falha no Qdrant Upsert: {operation_info}")
            
        return len(points)
