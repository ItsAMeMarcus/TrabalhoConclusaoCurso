from crewai import Agent, Task, Crew, Process
from app.workers.celery_app import celery_app
from app.workers.RAG.crew_agents import RAGAgents
from app.workers.RAG.crew_tasks import RAGTasks

# 4. Execução (Crew)

@celery_app.task
def run_crew_search(user_query):

    agentes = RAGAgents()
    tarefas = RAGTasks()

    
    buscador = agentes.buscador_vetores(query=user_query)
    formatador = agentes.formatador_resposta()

    tarefa_buscar = tarefas.tarefa_busca(buscador_vetores=buscador)
    tarefa_formatar = tarefas.tarefa_formatar(formatador_resposta=formatador, mensagem= user_query, tarefa_busca=tarefa_buscar)

    crew = Crew(
        agents=[buscador, formatador],
        tasks=[tarefa_buscar, tarefa_formatar],
        process=Process.sequential
    )
    result = crew.kickoff()
    return result.raw

# Exemplo de uso
# resposta_final = run_crew_search("Quais metodologias de NLP são usadas para análise jurídica?")
# print(resposta_final)
