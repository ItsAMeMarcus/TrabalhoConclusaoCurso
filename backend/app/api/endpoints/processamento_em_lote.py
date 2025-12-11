from typing import List
from fastapi import APIRouter, HTTPException
from app.workers.ETL.crew_assemble import processar_documento_pdf
import os
from pydantic import BaseModel
from celery import chain
# O 'router' é como um mini-aplicativo FastAPI que pode ser incluído no principal

router = APIRouter()

class DocumentoMetadata(BaseModel):
    titulo: str
    autor: str
    ano: int
    orientador: str
    caminho_arquivo: str

@router.post("/processar-lote-sequencial")
async def processar_lote_sequencial(lista_documentos: List[DocumentoMetadata]):
    """
    Lê arquivos locais e cria uma CORRENTE (Chain) de processamento.
    O PDF 2 só começa se o PDF 1 terminar com sucesso.
    Se o PDF 1 falhar, a corrente para e o PDF 2 nunca é processado.
    """
    
    # 1. Validação inicial: Checar se todos os arquivos existem antes de começar
    # Isso evita começar o lote se já sabemos que o 10º arquivo está faltando.
    for doc in lista_documentos:
        if not os.path.exists(doc.caminho_arquivo):
            raise HTTPException(
                status_code=400, 
                detail=f"Lote recusado. Arquivo não encontrado: {doc.caminho_arquivo}"
            )

    assinaturas_tarefas = []

    # 2. Montagem das Assinaturas das Tarefas
    for doc in lista_documentos:
        try:
            # Lê o binário do arquivo
            with open(doc.caminho_arquivo, "rb") as f:
                pdf_bytes = f.read()
            
            # Prepara os metadados
            metadados = doc.model_dump()
            metadados["filename_original"] = os.path.basename(doc.caminho_arquivo)
            del metadados["caminho_arquivo"]

            # CRUCIAL: Usamos .si() (Signature Immutable)
            # O .si() diz: "Execute esta tarefa com ESTES parâmetros fixos, 
            # e ignore o resultado da tarefa anterior, mas espere ela terminar com sucesso".
            task_signature = processar_documento_pdf.si(
                file_content=pdf_bytes, 
                metadados_formulario=metadados
            )
            
            assinaturas_tarefas.append(task_signature)

        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Erro ao ler arquivo local: {e}")

    if not assinaturas_tarefas:
        return {"status": "Nenhum arquivo para processar."}

    # 3. Criação e Execução da Chain
    # O chain conecta as tarefas: T1 -> T2 -> T3
    workflow = chain(*assinaturas_tarefas)
    
    # Dispara a execução assíncrona
    resultado_chain = workflow.apply_async()

    return {
        "status": "Lote sequencial iniciado.",
        "modo": "Chain (O próximo só inicia se o anterior tiver sucesso)",
        "quantidade": len(assinaturas_tarefas),
    }
