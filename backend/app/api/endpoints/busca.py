from typing import Annotated
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from app.workers.RAG.crew_assemble import run_crew_search 
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import torch

from app.workers.RAG.crew_assemble import run_crew_search

# O 'router' é como um mini-aplicativo FastAPI que pode ser incluído no principal
router = APIRouter()

@router.post("/busca-semantica")
async def criar_tarefa_processamento(
        query: Annotated[str, Form(...)],
        ):

    # O .delay() não espera a tarefa terminar, ele apenas a coloca na fila.
#    task = processar_documento_pdf.delay(file_content=pdf_bytes, metadados_formulario=metadados)

#    query = query
 #   qdrant_client = QdrantClient(host="localhost", port=6333)
  #  model_name = 'intfloat/multilingual-e5-large'
   # device = "cuda" if torch.cuda.is_available() else "cpu"
    #embedding_model = HuggingFaceEmbeddings(
     #   model_name=model_name,
      #  model_kwargs={'device': device}
#    )
    query_usuario = "query:" + query
    task = run_crew_search.delay(query_usuario)
        
 #   query_vector = embedding_model.embed_query(query)

  #  results = qdrant_client.query_points(
   #     collection_name="my_tcc_collection-com-mpnet-cosseno",
    #    query=query_vector,
     #   limit=5,
      #  with_payload=True,
       # with_vectors=False
#    )
    # Retorna uma resposta imediata para o cliente com o ID da tarefa
    return {f"status": "Rodando"}

