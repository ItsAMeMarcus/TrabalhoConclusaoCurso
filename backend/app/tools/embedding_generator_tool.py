import json
import torch
import os
from crewai.tools import BaseTool 
from typing import  Any
from pydantic.v1 import PrivateAttr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingGeneratorTool(BaseTool):
    name: str = "Ferramenta de Geração e Armazenamento de Embeddings"
    description: str = (
        "Recebe um dicionário JSON de chunks, gera seus embeddings localmente com BERT, "
        "e os armazena diretamente no banco de dados vetorial. Retorna uma mensagem de sucesso."
    )

    # --- Atributos privados para carregar o modelo uma única vez ---
    _embedding_model: Any = PrivateAttr()
    _vector_store: Any = PrivateAttr()
    _index_path: str = PrivateAttr()

    def __init__(self, index_path: str = "faiss_index", model_name='neuralmind/bert-base-portuguese-cased', **kwargs):
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

        # --- 2. Carrega ou Cria o Índice FAISS ---
        self._index_path = index_path
        try:
            if os.path.exists(index_path):
                self._vector_store = FAISS.load_local(index_path, self._embedding_model, allow_dangerous_deserialization=True)
                print(f"BertFaissStorageTool: Índice FAISS carregado de '{index_path}'.")
            else:
                # Se não existir, cria um FAISS vazio com um texto de placeholder
                print("BertFaissStorageTool: Nenhum índice FAISS encontrado. Criando um novo...")
                self._vector_store = FAISS.from_texts(["Iniciando o Vector Store."], self._embedding_model)
                self._vector_store.save_local(self._index_path)
                print(f"BertFaissStorageTool: Novo índice FAISS criado em '{index_path}'.")
        except Exception as e:
            print(f"Erro ao carregar/criar índice FAISS: {e}. Criando um novo em memória.")
            # Fallback para um índice em memória se tudo der errado
            self._vector_store = FAISS.from_texts(["Iniciando o Vector Store."], self._embedding_model)

    def _run(self, json_chunks_dict: str) -> str:
        """
        Método principal que recebe uma STRING JSON contendo chunks,
        gera os embeddings e retorna uma nova STRING JSON com os embeddings.
        """
        try:
            # --- MUDANÇA PRINCIPAL (ENTRADA) ---
            # 1. Converte a string JSON de volta para um dicionário Python
            chunks_dict = json.loads(json_chunks_dict)
            
            # 2. Extrai apenas os textos (values) do dicionário para processamento
            chunks_texts = list(chunks_dict.values())
            
            if not chunks_texts:
                return "Erro: O dicionário de chunks recebido está vazio."

            # O resto da lógica de embedding continua a mesma...
            print(f"[INFO TOOL]: Gerando e adicionando {len(chunks_texts)} chunks ao FAISS...")
            self._vector_store.add_texts(
                texts=chunks_texts
            )

            # --- ETAPA 3: Salvar o Índice Atualizado no Disco ---
            self._vector_store.save_local(self._index_path)
            print(f"[INFO TOOL]: Índice FAISS atualizado e salvo em '{self._index_path}'.")

            # --- ETAPA 4: Retornar Mensagem Simples (A Solução do Problema) ---
            return f"SUCESSO: {len(chunks_texts)} chunks foram vetorizados com Bertimbau e armazenados com sucesso no índice FAISS."

        except json.JSONDecodeError:
            return "Erro: A entrada não era uma string JSON válida."
        except Exception as e:
            return f"FALHA: Ocorreu um erro no processamento FAISS. Erro: {e}"

#testar isso depois
# from langchain_community.document_loaders import PyPDFLoader

# # 1. Instancie o Loader com o caminho do seu arquivo
# loader = PyPDFLoader("seu_documento.pdf")

# # 2. Use o método load() para extrair o texto em uma lista de 'Documents'
# documents = loader.load() 

# # 'documents' agora é uma lista, onde cada elemento é uma página do PDF
# print(f"Número de páginas carregadas: {len(documents)}")

# # Acessando o texto da primeira página
# primeira_pagina_texto = documents[0].page_content 
# print("\nConteúdo da Primeira Página (Primeiros 200 caracteres):")
# print(primeira_pagina_texto[:200])
