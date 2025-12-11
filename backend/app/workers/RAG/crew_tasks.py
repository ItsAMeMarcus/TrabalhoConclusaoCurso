from crewai import Task

from app.tools.RAG.vector_search_tool import SearchOutput

class RAGTasks():

    def tarefa_busca(self, buscador_vetores):
       return Task(
            description=f"Use a ferramenta 'Ferramenta de Extração de Texto' e passe o resultado para o próximo agente sem nenhum tipo de mudança",
            expected_output="Lista de chunks com metadados.",
            agent=buscador_vetores,
            output_json=SearchOutput
        )

    def tarefa_formatar(self, formatador_resposta, tarefa_busca, mensagem):
        return Task(
            description=(
                f"Com base nos resultados da busca, responda à pergunta do usuário: '{mensagem}'.\n"
                "Siga ESTRITAMENTE estas regras de apresentação para CADA TCC diferente encontrado:\n\n"
        
                "1. INTRODUÇÃO: Comece dizendo 'Aqui estão os trabalhos de conclusão de curso que abordam sua dúvida sobre [tema]...'\n"
        
                "2. ESTRUTURA POR TCC (Repita para cada obra relevante encontrada):\n"
                "   - Inicie o parágrafo exatamente com este modelo: "
                "   'O trabalho \"{titulo}\" foi escrito por {autor} e orientado por {orientador} em {ano}.'\n"
                "   - Na frase seguinte, conecte o conteúdo: "
                "   'Nesse trabalho, ele aborda [explicação baseada no TRECHO RELEVANTE encontrado]...'\n"
                "   - Use o texto do chunk para explicar como aquele TCC específico trata o tema da pergunta.\n\n"
        
                "3. AGRUPAMENTO: Se houver vários chunks do MESMO TCC, combine-os em um único parágrafo de explicação para não repetir a apresentação do autor.\n"
        
                "4. TOM DE VOZ: Seja útil, claro e encorajador."
            ),
            expected_output="Uma resposta em texto corrido seguindo o template de apresentação de autor/orientador/ano para cada TCC citado.",
            agent=formatador_resposta,
            context=[tarefa_busca]
        )
