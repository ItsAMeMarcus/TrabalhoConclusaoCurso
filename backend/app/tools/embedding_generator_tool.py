import json
from pydantic import BaseModel,Field
import torch
from crewai.tools import BaseTool 
from typing import  Any, Dict, List, Type
from pydantic.v1 import PrivateAttr
from langchain_huggingface import HuggingFaceEmbeddings
from app.data.vector_store import TccVectorStore

class ChunksInput(BaseModel):
    json_chunks: List[Dict[str,str]] = Field(description="Uma string JSON contendo a lista de chunks no formato '{ \"json_chunks\": [...] }'."
    )

class EmbeddingGeneratorTool(BaseTool):
    name: str = "Ferramenta de Geração e Armazenamento de Embeddings"
    description: str = (
        "Recebe um dicionário JSON de chunks, gera seus embeddings localmente com BERT, "
        "e os armazena diretamente no banco de dados vetorial. Retorna uma mensagem de sucesso."
    )
    args_schema: Type[BaseModel] = ChunksInput
    # --- Atributos privados para carregar o modelo uma única vez ---
    _embedding_model: Any = PrivateAttr()
    _metadata_tcc: Dict = PrivateAttr()
    _vector_store: TccVectorStore = PrivateAttr()  

    def __init__(self, metadados_tcc: Dict, model_name='neuralmind/bert-base-portuguese-cased', **kwargs):
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
            self._vector_store = TccVectorStore(vector_size=768)
            print("Preparações compĺetas, segue o baile")
        except Exception as e:
            print(f"Erro na conexão: {e}")

    def _run(self, json_chunks: List[Dict[str,str]]) -> str:
        """
        Método principal que recebe uma STRING JSON contendo chunks,
        gera os embeddings e retorna uma nova STRING JSON com os embeddings.
        """
        try:
            # --- MUDANÇA PRINCIPAL (ENTRADA) ---
            # 1. Converte a string JSON de volta para um dicionário Python
            # 2. Extrai apenas os textos (values) do dicionário para processamento
            # 1. Fazer o parse da string recebida
            # 2. Extrair a lista de chunks de dentro do JSON

            if not json_chunks:
                return "Erro: A string JSON não continha a chave 'json_chunks' ou a lista estava vazia."

            chunks_texts = []
            for chunk_dict in json_chunks:
                # chunk_dict é {"1": "Texto..."}
                # Precisamos extrair o valor "Texto..."
                if chunk_dict: # Garante que não está vazio
                    # Pega o primeiro (e único) valor de cada dicionário
                    texto_do_chunk = list(chunk_dict.values())[0]
                    chunks_texts.append(texto_do_chunk)
            
            if not chunks_texts:
                return "Erro: O dicionário de chunks recebido está vazio."

            # O resto da lógica de embedding continua a mesma...
            print(f"[INFO TOOL]: Gerando {len(chunks_texts)} chunks...")
            embeddings_vetoriais = self._embedding_model.embed_documents(chunks_texts)
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
            qtd_salva = self._vector_store.upsert_chunks(dados_qdrant)
            # --- ETAPA 4: Retornar Mensagem Simples (A Solução do Problema) ---

            return f"Sucesso: gerados {qtd_salva} chunks e salvos no banco de dados!"
                    
        except json.JSONDecodeError:
            return "Erro: A entrada não era uma string JSON válida."
        except ValueError as ve:
            # Erro de dimensão (384 vs 768)
            return f"ERRO DE CONFIGURAÇÃO: {str(ve)}"
        except Exception as e:
            # Outros erros
            print(f"Erro detalhado na Tool: {e}")
            return f"ERRO NO PROCESSAMENTO: {str(e)}"
