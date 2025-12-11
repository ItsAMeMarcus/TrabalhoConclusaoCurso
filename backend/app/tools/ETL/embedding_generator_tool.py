import json
from pydantic import BaseModel,Field
import torch
from crewai.tools import BaseTool 
from typing import  Any, Dict, List, Type
from pydantic import PrivateAttr
from langchain_huggingface import HuggingFaceEmbeddings
from app.data.vector_store import TccVectorStore
from qdrant_client.http.models import Distance

class ChunksInput(BaseModel):
    json_chunks: List[Dict[str,str]] = Field(description="Uma string JSON contendo a lista de dicionário de chunks."
    )

class EmbeddingGeneratorTool(BaseTool):
    name: str = "Ferramenta de Geração e Armazenamento de Embeddings"
    description: str = (
        "Recebe uma lista de dicionários JSON com chunks, gera seus embeddings localmente com BERT, "
        "e os armazena diretamente no banco de dados vetorial. Retorna uma mensagem de sucesso."
    )
    args_schema: Type[BaseModel] = ChunksInput
    # --- Atributos privados para carregar o modelo uma única vez ---
    _embedding_model:Any = PrivateAttr(default=None)
    _metadata_tcc:Any = PrivateAttr(default=None)
    _vector_store_dot:Any = PrivateAttr(default=None)  
    _vector_store_cosine:Any = PrivateAttr(default=None)  
    _vector_store_euclidiana:Any = PrivateAttr(default=None)  

    def __init__(self, metadados_tcc: Dict, model_name='intfloat/multilingual-e5-large', **kwargs):
        super().__init__(**kwargs)
        
        # --- 1. Inicializa o Modelo de Embedding (Bertimbau) ---
        # Usamos o wrapper do LangChain para ser compatível com o FAISS.
        # Ele vai rodar localmente no seu dispositivo.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"BertFaissStorageTool: Carregando modelo '{model_name}' no dispositivo: {device}")
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        print(f"BertFaissStorageTool: Modelo '{model_name}' carregado.")

        self._metadados_tcc = metadados_tcc
        try:
            self._vector_store_cosine = TccVectorStore(collection_name="tcc-collection-cosseno", distancia=Distance.COSINE)
            self._vector_store_dot = TccVectorStore(collection_name="tcc-collection-ponto", distancia=Distance.DOT)
            self._vector_store_euclidiana= TccVectorStore(collection_name="tcc-collection-euclidiana", distancia=Distance.EUCLID)
            print("Preparações compĺetas, segue o baile")
        except Exception as e:
            print(f"Erro na conexão: {e}")

    def _run(self, json_chunks: List[Dict[str,str]]) -> str:
        """
        Método principal que recebe uma STRING JSON contendo chunks,
        gera os embeddings e retorna uma nova STRING JSON com os embeddings.
        """
        try:

            if not json_chunks:
                return "Erro: A string JSON não continha a chave 'json_chunks' ou a lista estava vazia."

            chunks_texts = []
            for chunk_dict in json_chunks:
                if chunk_dict: # Garante que não está vazio
                    texto_do_chunk = list(chunk_dict.values())[0]
                    texto_do_chunk = f"passage: {texto_do_chunk}"
                    chunks_texts.append(texto_do_chunk)
            
            if not chunks_texts:
                return "Erro: O dicionário de chunks recebido está vazio."

            print(f"[INFO TOOL]: Gerando {len(chunks_texts)} chunks...")
            embeddings_vetoriais = self._embedding_model.embed_documents([f"passage: {c}" for c in chunks_texts])
            print("Embeddings gerados.")

            dados_qdrant= []
            for i, texto in enumerate(chunks_texts):
                dados_qdrant.append({
                    "text": texto,
                    "embedding": embeddings_vetoriais[i],
                    "metadata": {
                        **self._metadados_tcc,
                        "ordem_chunk": i
                    }
                })

            print("Mandando chunks para o Qdrant...")
            qtd_salva_cosine = self._vector_store_cosine.upsert_chunks(dados_qdrant)
            qtd_salva_dot = self._vector_store_dot.upsert_chunks(dados_qdrant)
            qtd_salva_euclid = self._vector_store_euclidiana.upsert_chunks(dados_qdrant)

            return f"Sucesso: gerados {qtd_salva_cosine} chunks de cosseno, {qtd_salva_euclid} de euclidiano e {qtd_salva_dot} de ponto salvos no banco de dados!"
                    
        except json.JSONDecodeError:
            return "Erro: A entrada não era uma string JSON válida."
        except ValueError as ve:
            # Erro de dimensão (384 vs 768)
            return f"ERRO DE CONFIGURAÇÃO: {str(ve)}"
        except Exception as e:
            # Outros erros
            print(f"Erro detalhado na Tool: {e}")
            return f"ERRO NO PROCESSAMENTO: {str(e)}"
