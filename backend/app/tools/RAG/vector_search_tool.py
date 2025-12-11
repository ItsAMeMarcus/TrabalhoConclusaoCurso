from typing import Dict, List
from pydantic import BaseModel,Field
import torch
import json
from crewai.tools import BaseTool 
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import NamedVector

class SearchOutput(BaseModel):
    json_search: List[Dict[str,str]] = Field(
        description="Uma lista contendo todos os resultados de uma busca ao banco"
    )

class VectorSearchInput(BaseModel):
    argument: str = Field(..., description="Lê um PDF e extrai o texto dele, retornando-o em uma string e passando para os próximos agentes")

class VectorSearchTool(BaseTool):

    name: str = "Ferramenta de Extração de Texto"
    description: str = "Lê um PDF e extrai o texto dele, retornando-o em uma string e passando para os próximos agentes"
    
    def __init__(self, query:str, model_name= 'intfloat/multilingual-e5-large',**kwargs):
        super().__init__(**kwargs)
        self._query = query
        self._qdrant_client = QdrantClient(host="localhost", port=6333)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
    def _run(self) -> str | None:
        """
        Útil para buscar informações técnicas, metodologias e resultados em uma base de TCCs e Artigos.
        Retorna trechos relevantes com fonte e página.
        """
        # Uso de Embeddings melhores (Visão Data Science - Cersei)
        query_vector = self._embedding_model.embed_query(self._query)

        results = self._qdrant_client.query_points(
            collection_name="my_tcc_collection-com-mpnet-euclidiano",
            query=query_vector,
            limit=10,
            with_payload=True,
            with_vectors=False
        )
    
        try:
            context = []
            for r in results.points:
                if r.payload is not None:
                    filename = r.payload.get('filename_original')
                    autor = r.payload.get('autor')
                    orientador = r.payload.get('orientador')
                    titulo = r.payload.get('titulo')
                    conteudo = r.payload.get('texto_chunk', '')

                    entry = {
                        "filename": filename,
                        "autor": autor,
                        "orientador":orientador,
                        "titulo":titulo,
                        "conteudo":conteudo
                    }
                    context.append(entry)
            final_output_dict = {"json_search": context}

            return json.dumps(final_output_dict, ensure_ascii=False)
        except Exception as e:
            return f"Erro ao retornar o resultado em um dicionário: {e}"
