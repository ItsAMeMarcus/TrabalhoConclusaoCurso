from .celery_app import celery_app
import PyPDF2
import io
from app.core.config import llm_rapido, llm_pro, langfuse

import langchain
from langchain.globals import set_debug

# --- ADICIONE ESTAS DUAS LINHAS AQUI ---
set_debug(True)

from langfuse.langchain import CallbackHandler
# --- IMPORTS PADRÃO E DE SISTEMA ---
import os
import json
from typing import List, Dict, Any

# --- IMPORTS DE FERRAMENTAS ---
from app.tools.semantic_splitter_tool import SemanticTextSplitterTool
from app.tools.embedding_generator_tool import EmbeddingGeneratorTool
from app.tools.text_splitter_tool import TextSplitterTool
# from app.tools.vector_db_tool import EmbeddingGeneratorTool

# --- IMPORTS DO CREWAI ---
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

os.environ["OPENAI_API_KEY"] = "NA" 

# --- IMPORTS DE BANCO DE DADOS ---
import psycopg2
from psycopg2.extras import execute_values # Otimização para inserção em lote

# --- IMPORTS PARA GERENCIAMENTO DE MODELOS E CONFIGURAÇÃO ---
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic.v1 import PrivateAttr
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

    langfuse_handler = CallbackHandler()
        

    # 1. Ler o conteúdo do arquivo    
    texto_completo = _read_pdf_text(file_content)
    
    if not texto_completo:
        print("PDF vazio ou não foi possível extrair texto. Finalizando a tarefa.")
        return "Falha na extração de texto."
    # 2. Definir Agentes, Ferramentas e Tarefas
    #    (Todo o setup do seu notebook vem para cá)
    # Agente 1: Responsável por dividir o texto
    # 
    text_splitter = TextSplitterTool()
    semantic_splitter = SemanticTextSplitterTool()
    embedding_generator = EmbeddingGeneratorTool()

    preparador_texto = Agent(
        role='Preparador de Conteúdo',
        goal='Ler textos grandes e passar para análise subsequente.',
        backstory=(
            "Você é um assistente de IA especializado em processamento e preparação de dados textuais. "
            "Sua principal habilidade é fazer a leitura de documentos grandes, "
            "garantindo que o texto esteja pronto para ser processado por outros agentes."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm_rapido
    )

    # Agente 2: Responsável por revisar cada pedaço
    revisor_conteudo = Agent(
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
        llm=llm_pro
    )

    # Agente 3: Responsável por corrigir o texto
    corretor_feedback = Agent(
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
    segmentador = Agent(
        role='Segmentador Semântico de Documentos',
        goal='Dividir um documento de texto limpo em chunks coesos e semanticamente significativos, preparando-os para a vetorização.',
        backstory=(
            "Você é um especialista em pré-processamento para modelos de linguagem. Você entende que a qualidade dos "
            "embeddings vetoriais depende diretamente da qualidade dos chunks de texto. Sua principal habilidade é "
            "identificar as fronteiras naturais de um texto — como parágrafos e sentenças — para criar segmentos "
            "que encapsulem ideias completas, garantindo que cada vetor gerado posteriormente represente um conceito "
            "claro e distinto."
        ),
        tools=[semantic_splitter], # Equipado com a ferramenta semântica
        allow_delegation=False,
        verbose=True,
        # Pode usar um LLM rápido, pois a tarefa é principalmente chamar a ferramenta
        llm=llm_pro
    )

    # --- AGENTE 5: GERADOR DE EMBEDDINGS ---
    gerador_embeddings = Agent(
        role='Especialista em Vetorização de Texto',
        goal='Converter chunks de texto limpo em representações numéricas (embeddings).',
        backstory=(
            "Você é um engenheiro de Processamento de Linguagem Natural (PNL) focado em transformar a linguagem humana "
            "num formato que os computadores possam entender. Sua especialidade é usar modelos de deep learning, como o BERT, "
            "para capturar o significado semântico profundo de um texto. Você não lê apenas as palavras, você compreende o "
            "contexto e a intenção por trás delas, convertendo essa compreensão em vetores numéricos de alta dimensão. "
            "Seu trabalho é a ponte entre o texto qualitativo e a análise quantitativa, preparando o terreno para "
            "tarefas avançadas como busca por similaridade, classificação de documentos e clustering de tópicos."
        ),
        llm=llm_rapido,
        tools=[embedding_generator],
        allow_delegation=False,
        verbose=True
    )

    # Tarefa 1: Dividir o texto
    tarefa_leitura = Task(
        description=f"""
        Leia o texto e passe para o próximo agente.

        Aqui está o texto que você precisa passar:

        {texto_completo}
        """,
        expected_output='O texto a ser revisado, pronto para ser enviado para o revisor.',
        agent=preparador_texto
    )

    # Tarefa 2: Revisar cada pedaço
    # O 'context' aqui é crucial. Ele pega o output da tarefa anterior.
    tarefa_revisar = Task(
        description= f"""Realize uma revisão detalhada. Identifique todos os erros de gramática, ortografia, pontuação, clareza e formatação.
            ATENÇÃO ESPECIAL: LIMPEZA DE TEXTO DE GRÁFICOS.
            O texto pode conter sequências de palavras que foram extraídas de gráficos, diagramas ou fluxogramas (vetores no PDF). Essas sequências não formam frases coerentes e devem ser removidas.
            Identifique esses trechos e, no seu relatório, instrua o corretor a removê-los completamente.

            EXEMPLOS DE TEXTO PROVENIENTE DE GRÁFICO A SER REMOVIDO:
            - '57% 43% Gênero. Masculino. Feminino.'
            - '20h 30h 40h 44h CR x Jornada de trabalho. '

            IMPORTANTE: Legendas de figuras (ex: 'Figura 7. Sistemas e dos dados necessários...') geralmente são frases válidas e devem ser mantidas, mas o conjunto de rótulos desconexos que às vezes as acompanha deve ser marcado para remoção.
            Em seguida, compile um relatório consolidado com todas as sugestões. IMPORTANTE: Não corrija o texto diretamente. Seu resultado final deve ser apenas a lista de feedbacks.
            """,
        expected_output=(
            "Um único documento de texto em formato JSON. O JSON deve ser uma lista de objetos, "
            "onde cada objeto representa um erro e contém duas chaves: 'trecho_original' e 'feedback_detalhado'.\n\n"
            "Exemplo de Saída:\n"
            '[\n'
            '  {\n'
            '    "trecho_original": "...o sistema se base ada...",\n'
            '    "feedback_detalhado": "Corrigir \'base ada\' para \'baseia\'. Remover a frase \'MySQL Sistema...\' pois é um artefato de gráfico."\n'
            '  },\n'
            '  {\n'
            '    "trecho_original": "...a borda o conceito...",\n'
            '    "feedback_detalhado": "Corrigir \'a borda\' para \'aborda\'."\n'
            '  }\n'
            ']'
        ),
        agent=revisor_conteudo,
    )

    # Tarefa 3: Receber os feedbacks e corrigir os textos
    tarefa_corrigir = Task(
        description=(
            "Você recebeu dois insumos: o texto original e o relatório de feedbacks do revisor. "
            "Sua missão é processar uma lista de trechos do texto e seus respectivos feedbacks e aplicar cada uma das sugestões de correção ao texto. "
            "Seu resultado final deve ser o texto corrigido e limpo."
            "APLIQUE TODAS AS CORREÇÕES SUGERIDAS A TODOS OS TRECHOS. "
            "O seu resultado final deve ser o texto, com todos os trechos já corrigidos, formatado para NLP. "
            "Tente ser o mais eficiente possível, processando o máximo de trechos que conseguir em cada passo do seu raciocínio."
        ),
        expected_output=(
            "Sua missão tem duas fases críticas:\n\n"
            "**Fase 1: Correção do Texto**\n"
            "Você recebeu dois insumos: o texto original e um relatório de feedbacks do revisor que tem os trechos a serem corrigidos. "
            "Primeiro, aplique CADA uma das sugestões de correção e remoção aos seus respectivos pedaços de texto. "
            "Seja extremamente fiel ao feedback fornecido.\n\n"
            "**Fase 2: Formatação para NLP (BERT)**\n"
            "Após aplicar todas as correções, você deve realizar uma formatação final nos textos. "
            "O objetivo é prepará-lo para ser usado em tarefas de NLP. Siga estas regras rigorosamente:\n"
            "1. **Estrutura de Parágrafos:** Garanta que haja apenas UMA ÚNICA quebra de linha (\\n) para separar os parágrafos. Remova quebras de linha duplas ou triplas.\n"
            "2. **Fluxo das Frases:** Elimine todas as quebras de linha que ocorrem no MEIO de uma frase, unindo o texto para que cada frase seja uma linha contínua dentro do seu parágrafo.\n"
            "3. **Limpeza de Espaços:** Remova todos os espaços em branco desnecessários, como espaços duplos entre palavras ou espaços no início ou fim de uma linha.\n"
            "4. **Remoção Final de Artefatos:** Faça uma varredura final para remover quaisquer artefatos restantes que não sejam texto corrido (ex: números de página)."
        ),
        agent=corretor_feedback,
        context=[tarefa_revisar, tarefa_leitura] # Usa o resultado da tarefa de revisão
    )

    #Tarefa 4: Dividir o texto já corrigido sem perder o sentido.
    tarefa_segmentar = Task(
        description=(
            "Você recebeu um texto já pré-corrigido. "
            "Sua única e exclusiva missão é pegar ess texto e passá-lo diretamente como o argumento 'text_content' para a ferramenta 'Ferramenta de Divisão Semântica de Texto'. "
            "Use a 'Ferramenta de Divisão Semântica de Texto' para dividir este texto completo "
            "em chunks que sejam semanticamente coesos e otimizados."
            "Não tente analisar ou modificar a string, apenas a repasse para a ferramenta e passe seu resultado para o próximo agente."
        ),
        expected_output=(
        "Uma ÚNICA STRING em formato JSON que representa um dicionário que vai vir diretamente da ferramenta. "
        "As chaves do dicionário vão ser os números dos chunks em ordem cronológica (começando em '1'), "
        "e os valores vão ser os textos dos chunks. "
        "Exemplo: '[{1: \\\"Pedaco1\\\"}, {2: \\\"Pedaco2\\\"}, {3: \\\"Pedaco3\\\"}]'"
    ),
        agent=segmentador,
        # Esta tarefa precisa do output do corretor para começar.
        context=[tarefa_corrigir]
    )

    #Tarefa 5: Gerar embeddings
    tarefa_gerar_embeddings = Task(
        description=(
        "Sua única tarefa é pegar o resultado da tarefa anterior, que é uma string JSON de um dicionário de chunks, "
        "e passá-lo como o único argumento para a ferramenta 'Gerador de Embeddings BERT'. "
        "O resultado desta ferramenta deve ser sua resposta final, sem nenhuma palavra ou pensamento adicional."
        ),
        expected_output=(
            "A string JSON exata e bruta retornada pela ferramenta 'Gerador de Embeddings BERT'. "
            "Não inclua 'Final Answer:', pensamentos, ou qualquer outro texto. Apenas a string JSON."
        ),
        agent=gerador_embeddings,
        context=[tarefa_segmentar] # Recebe a lista de chunks da tarefa anterior
    )

    
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
    print(resultado)
    
    langfuse.flush()
    return json.loads(resultado)
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