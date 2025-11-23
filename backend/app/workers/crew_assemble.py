from .celery_app import celery_app
from langchain.globals import set_debug
from crewai import Crew, Process
from app.workers.crew_agents import PreProcessingAgents
from app.workers.crew_tasks import PreProcessingTasks

#os.environ["OPENAI_API_KEY"] = "NA" 
from dotenv import load_dotenv

# Carrega as variáveis de ambiente (como a GOOGLE_API_KEY) do arquivo .env
load_dotenv()
set_debug(True)

@celery_app.task
def processar_documento_pdf(file_content: bytes, metadados_formulario: dict):
    """
    Esta função é executada por um Worker do Celery em segundo plano.
    """
    # 1. Ler o conteúdo do arquivo    
    try:
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
        gerador_embeddings =agentes.gerador_embeddings(metadados_tcc=metadados_formulario)

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
        print("Processamento e armazenamento concluídos com sucesso!")
    
        return resultado.raw

    except Exception as e:
        print(f"ERRO ao processar TCC: {e}")
        return f"FALHA: {e}"
