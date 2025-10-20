from .celery_app import celery_app
import PyPDF2
import io
from app.core.config import langfuse

from langchain.globals import set_debug

# --- ADICIONE ESTAS DUAS LINHAS AQUI ---
set_debug(True)

# --- IMPORTS PADRÃO E DE SISTEMA ---
import os
import json

# --- IMPORTS DO CREWAI ---
from crewai import Crew, Process
from app.workers.crew_agents import PreProcessingAgents
from app.workers.crew_tasks import PreProcessingTasks

os.environ["OPENAI_API_KEY"] = "NA" 

# --- IMPORTS DE BANCO DE DADOS ---
import psycopg2
from psycopg2.extras import execute_values # Otimização para inserção em lote

# --- IMPORTS PARA GERENCIAMENTO DE MODELOS E CONFIGURAÇÃO ---
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (como a GOOGLE_API_KEY) do arquivo .env
load_dotenv()

def _read_pdf_text(pdf_bytes: bytes) -> str | None:
    """
    Abre um objeto de bytes de um PDF em memória e extrai seu texto.
    """
    try:
        # Cria um "arquivo virtual" em memória a partir dos bytes
        pdf_file_in_memory = io.BytesIO(pdf_bytes)
        
        reader = PyPDF2.PdfReader(pdf_file_in_memory)
        texto_completo = ""
        for page in reader.pages:
            texto_completo += page.extract_text() + "\n"
        return texto_completo
    except Exception as e:
        print(f"Erro ao ler o PDF a partir dos bytes: {e}")
        raise e

# O código do seu notebook vira uma função, marcada como uma tarefa do Celery
@celery_app.task
def processar_documento_pdf(file_content: bytes):
    """
    Esta função é executada por um Worker do Celery em segundo plano.
    """

    # 1. Ler o conteúdo do arquivo    
    texto_completo = _read_pdf_text(file_content)
    
    if not texto_completo:
        print("PDF vazio ou não foi possível extrair texto. Finalizando a tarefa.")
        return "Falha na extração de texto."
    # 2. Definir Agentes, Ferramentas e Tarefas
    agentes = PreProcessingAgents()
    tarefas = PreProcessingTasks()


    preparador_texto = agentes.preparador_texto()
    revisor_conteudo = agentes.revisor_conteudo()
    corretor_feedback = agentes.corretor_feedback()
    segmentador = agentes.segmentador()
    gerador_embeddings =agentes.gerador_embeddings() 

    # Tarefa 1: Dividir o texto
    tarefa_leitura = tarefas .tarefa_leitura(preparador_texto=preparador_texto,texto_completo=texto_completo)
    tarefa_revisar = tarefas.tarefa_revisar(tarefa_leitura=tarefa_leitura,revisor_conteudo=revisor_conteudo)
    tarefa_corrigir = tarefas.tarefa_corrigir(corretor_feedback=corretor_feedback,tarefa_leitura=tarefa_leitura,tarefa_revisar=tarefa_revisar) 
    tarefa_segmentar = tarefas.tarefa_segmentar(segmentador=segmentador,tarefa_corrigir=tarefa_corrigir)
    tarefa_gerar_embeddings = tarefas.tarefa_gerar_embeddings(gerador_embeddings=gerador_embeddings,tarefa_segmentar=tarefa_segmentar)
    
    # 3. Montar e executar a Crew
    crew = Crew(
        agents=[preparador_texto, revisor_conteudo, corretor_feedback, segmentador, gerador_embeddings],
        tasks=[tarefa_leitura, tarefa_revisar, tarefa_corrigir,tarefa_segmentar,tarefa_gerar_embeddings],
        process=Process.sequential,
        verbose=True
    )
    with langfuse.start_as_current_span(name="crewai-tcc-trace"):
        resultado = crew.kickoff()

    # 4. (Opcional) Salvar o resultado final no banco de dados ou em outro lugar
    print("Processamento concluído com sucesso!")
    print(resultado.raw)
    
    langfuse.flush()
    return json.loads(resultado.raw)
    ###IMPORTANTE###
    # - Podemos voltar aqui para estruturar a saída final da nossa equipe
    # from pydantic import BaseModel
    # from typing import List

    # class ResearchFindings(BaseModel):
    # main_points: List[str]
    # key_technologies: List[str]
    # future_predictions: str

    # # Get structured output
    # result = researcher.kickoff(
    # "Summarize the latest developments in AI for 2025",
    # response_format=ResearchFindings
    # )

    # # Access structured data
    # print(result.pydantic.main_points)
    # print(result.pydantic.future_predictions)
