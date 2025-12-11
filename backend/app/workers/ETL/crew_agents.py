from crewai import Agent
from app.core.config import llm_rapido, llm_pro, llm_flash, llm_flash_lite
from app.tools.ETL.embedding_generator_tool import EmbeddingGeneratorTool
from app.tools.ETL.pdf_loader import PDFLoaderTool
from app.tools.ETL.semantic_splitter_tool import SemanticTextSplitterTool

class PreProcessingAgents():

    def preparador_texto(self, pdf_bytes: bytes):
        return Agent(
            role='Preparador de Conteúdo',
            goal='Ler textos grandes e passar para análise subsequente.',
            backstory=(
                "Você é um assistente de IA especializado em processamento e preparação de dados textuais. "
                "Sua principal habilidade é fazer a leitura de documentos grandes, "
                "garantindo que o texto esteja pronto para ser processado por outros agentes."
            ),
            verbose=False,
            tools=[PDFLoaderTool(result_as_answer=True, pdf_bytes= pdf_bytes)],
            llm=llm_flash
        )

    # Agente 2: Responsável por revisar cada pedaço
    def revisor_conteudo(self):

        return Agent(
            role='Revisor de Conteúdo Sênior',
            goal='Analisar criticamente cada pedaço de texto, identificando erros gramaticais, de clareza e de estilo, e gerar um relatório de feedback acionável.',
            backstory=(
                "Você é um editor sênior com vasta experiência em textos acadêmicos. Seu olhar é treinado para detectar "
                "imperfeições, desde erros gramaticais simples até artefatos de extração de dados, como textos lidos de gráficos, remoção de tokens estranhos, espaços duplicados e por ai vai. "
                "Sua função não é reescrever o texto, mas sim fornecer instruções claras e precisas para que um corretor "
                "possa executar as alterações perfeitamente, garantindo que apenas prosa coerente permaneça no documento final."
            ),
            verbose=True,
            allow_delegation=False,
            llm=llm_rapido
        )

    # Agente 3: Responsável por corrigir o texto
    def corretor_feedback(self):
        return Agent(
            role='Corretor e Formatador de Texto para NLP',
            goal='Aplicar feedbacks de revisão a um texto e formatar o resultado final para ser otimizado para processamento por modelos de NLP como o BERT de forma eficiente.',
            backstory=(
                "Você é um especialista em processamento de texto com dupla função. Além de ser um corretor impecável que segue "
                "instruções à risca, você entende profundamente as necessidades de modelos de Processamento de Linguagem Natural (NLP). "
                "Sua especialidade é transformar textos revisados em um corpo de texto (corpus) limpo, estruturado e contínuo, "
                "ideal para tarefas como análise de sentimentos, reconhecimento de entidades e treinamento de modelos como o BERT, "
                "removendo qualquer ruído que possa prejudicar a análise computacional."
            ),
            verbose=True,
            allow_delegation=False,
            llm=llm_pro
        )

    #Agente 4: Responsável em dividir o texto focando em manter o valor semantico desses chunks
    def segmentador(self, texto_completo:str):
        return Agent(
            role='Segmentador Semântico de Documentos',
            goal='Dividir um documento de texto limpo em chunks coesos e semanticamente significativos, preparando-os para a vetorização.',
            backstory=(
                "Você é um especialista em pré-processamento para modelos de linguagem. Você entende que a qualidade dos "
                "embeddings vetoriais depende diretamente da qualidade dos chunks de texto. Sua principal habilidade é "
                "identificar as fronteiras naturais de um texto — como parágrafos e sentenças — para criar segmentos "
                "que encapsulem ideias completas, garantindo que cada vetor gerado posteriormente represente um conceito "
                "claro e distinto."
            ),
            tools=[SemanticTextSplitterTool(result_as_answer=True, text_content=texto_completo)], # Equipado com a ferramenta semântica
            allow_delegation=False,
            verbose=True,
            # Pode usar um LLM rápido, pois a tarefa é principalmente chamar a ferramenta
            llm=llm_flash
        )

    # --- AGENTE 5: GERADOR DE EMBEDDINGS ---
    def gerador_embeddings(self, metadados_tcc):
        return Agent(
            role='Especialista em Indexação de Documentos',
            goal='Receber chunks de texto, convertê-los em embeddings e armazená-los no PGVector para futura recuperação.',
            backstory=(
                "Você é um engenheiro de dados focado em eficiência. Sua função é estritamente técnica: "
                "executar a Tool de Geração de Embeddings com a maior economia de contexto possível, "
                "garantindo que APENAS o chunk seja passado para a Tool."
            ),          
            llm=llm_flash,
            tools=[EmbeddingGeneratorTool(metadados_tcc=metadados_tcc)],
            allow_delegation=False,
            verbose=False,
            max_iter=1
        )
