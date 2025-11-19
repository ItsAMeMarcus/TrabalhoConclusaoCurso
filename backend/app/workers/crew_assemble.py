from sqlalchemy.exc import SQLAlchemyError
from .celery_app import celery_app
#from app.core.config import langfuse
from app.data.models import Tcc
from app.data.database import SessionLocal
from langchain.globals import set_debug

# --- ADICIONE ESTAS DUAS LINHAS AQUI ---
set_debug(True)

from crewai import Crew, Process
from app.workers.crew_agents import PreProcessingAgents
from app.workers.crew_tasks import PreProcessingTasks

#os.environ["OPENAI_API_KEY"] = "NA" 
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (como a GOOGLE_API_KEY) do arquivo .env
load_dotenv()

@celery_app.task
def processar_documento_pdf(file_content: bytes, metadados_formulario: dict):
    """
    Esta função é executada por um Worker do Celery em segundo plano.
    """
    db = SessionLocal()
    # 1. Ler o conteúdo do arquivo    
    try:
        novo_tcc = Tcc(
            titulo=metadados_formulario.get("titulo"),
            ano=metadados_formulario.get("ano"),
            orientador=metadados_formulario.get("orientador"), 
            autor=metadados_formulario.get("autor")
        )

        #if not texto_completo:
         #   print("PDF vazio ou não foi possível extrair texto. Finalizando a tarefa.")
          #  return "Falha na extração de texto."
        # 2. Definir Agentes, Ferramentas e Tarefas
        agentes = PreProcessingAgents()
        tarefas = PreProcessingTasks()


        preparador_texto = agentes.preparador_texto(pdf_bytes=file_content)

        revisor_conteudo = agentes.revisor_conteudo()
        corretor_feedback = agentes.corretor_feedback()

        tarefa_leitura = tarefas .tarefa_leitura(preparador_texto=preparador_texto)
        tarefa_revisar = tarefas.tarefa_revisar(tarefa_leitura=tarefa_leitura,revisor_conteudo=revisor_conteudo)
        tarefa_corrigir = tarefas.tarefa_corrigir(corretor_feedback=corretor_feedback,tarefa_leitura=tarefa_leitura,tarefa_revisar=tarefa_revisar) 
    
        # 3. Montar e executar a Crew
        crew_limpeza = Crew(
            agents=[preparador_texto, revisor_conteudo, corretor_feedback],
            tasks=[tarefa_leitura, tarefa_revisar, tarefa_corrigir],
            process=Process.sequential,
            verbose=False,
            tracing=True,
            memory=False,
            max_rpm=10
        )
        texto_completo = crew_limpeza.kickoff().raw

        segmentador = agentes.segmentador(texto_completo=texto_completo)
        gerador_embeddings =agentes.gerador_embeddings(novo_tcc=novo_tcc, db=db) 

        tarefa_segmentar = tarefas.tarefa_segmentar(segmentador=segmentador,tarefa_corrigir=tarefa_corrigir)
        tarefa_gerar_embeddings = tarefas.tarefa_gerar_embeddings(gerador_embeddings=gerador_embeddings,tarefa_segmentar=tarefa_segmentar)

        crew_embedding = Crew(
            agents=[segmentador,gerador_embeddings],
            tasks=[tarefa_segmentar,tarefa_gerar_embeddings],
            process=Process.sequential,
            verbose=False,
            tracing=True,
            memory=False,
            max_rpm=10
        )

        resultado = crew_embedding.kickoff()
        print("Processamento concluído com sucesso!")
    
        db.commit()
        return resultado.raw

    except SQLAlchemyError as e:
        db.rollback()
        print(f"Deu erro no banco: {e}")
    except Exception as e:
        db.rollback() 
        print(f"ERRO ao processar TCC: {e}")
        return f"FALHA: {e}"
    finally:
        db.close() #
