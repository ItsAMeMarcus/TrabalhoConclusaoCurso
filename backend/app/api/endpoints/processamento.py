import shutil
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.workers.crew_assemble import processar_documento_pdf

# O 'router' é como um mini-aplicativo FastAPI que pode ser incluído no principal
router = APIRouter()

@router.post("/processar-documento")
async def criar_tarefa_processamento(file: UploadFile = File(...)):
    """
    Recebe um arquivo PDF, lê seu conteúdo para a memória e inicia a tarefa
    de processamento em segundo plano, passando os bytes do arquivo.
    """
    # if file.content_type != "application/pdf":
    #     raise HTTPException(status_code=400, detail="Tipo de arquivo inválido. Apenas PDFs são aceitos.")
    # # Salva o arquivo enviado no disco
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Não foi possível ler o arquivo: {e}")
    finally:
        await file.close()


    # O .delay() não espera a tarefa terminar, ele apenas a coloca na fila.
    task = processar_documento_pdf.delay(file_content=pdf_bytes)

    # Retorna uma resposta imediata para o cliente com o ID da tarefa
    return {"job_id": task.id, "status": "Tarefa de processamento iniciada com sucesso."}
