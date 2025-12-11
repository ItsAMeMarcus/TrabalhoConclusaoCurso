from crewai import Agent
from app.core.config import llm_rapido, llm_pro, llm_flash, llm_flash_lite
from app.tools.RAG.vector_search_tool import VectorSearchTool

class RAGAgents():
    
    def buscador_vetores(self, query:str):
        return Agent(
            role='Bibliotecário de TCCs',
            goal='Encontrar os TCCs exatos que contêm a resposta para a pergunta do usuário no Qdrant.',
            backstory="Você é eficiente e preciso. Seu trabalho é apenas recuperar os dados brutos e metadados.",
            tools=[VectorSearchTool(query=query)],
            llm=llm_flash,
            verbose=True,
            allow_delegation=False
        )

    def formatador_resposta(self):
        return Agent(
            role='Consultor Acadêmico',
            goal='Responder ao aluno apresentando os TCCs encontrados de forma narrativa e estruturada.',
            backstory=(
                "Você é um orientador acadêmico que ajuda alunos a encontrar referências. "
                "Você não joga informações soltas. Você apresenta cada trabalho com cordialidade, "
                "dando crédito aos autores e explicando o conteúdo encontrado."
            ),
            llm=llm_flash,
            verbose=True
        )
