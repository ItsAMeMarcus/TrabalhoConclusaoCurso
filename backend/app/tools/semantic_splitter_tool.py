import json
from crewai.tools import BaseTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List,Dict


class ChunkItem(BaseModel):
    # Usaremos Dict[str, str] para simplificar, já que a maioria dos parsers JSON trata chaves numéricas como strings.
    # O LLM é instruído a outputar o formato exato.
    pass 

class ChunksOutput(BaseModel):
    """Estrutura final para a saída de segmentação de texto."""
    # O Field deve ser uma lista de dicionários, onde cada dicionário representa um chunk.
    json_chunks: List[Dict[str, str]] = Field(
        description="Uma lista contendo todos os chunks do texto, onde a chave é o número do chunk (como string) e o valor é o conteúdo do chunk."
    )

class SemanticTextSplitterInput(BaseModel):
    argument: str = Field(..., description="Divide um texto longo em chunks, retornando um JSON que contém um dicionario com os chunks enumerados por ordem")

class SemanticTextSplitterTool(BaseTool):
    name: str = "Ferramenta de Divisão Semântica de Texto"
    description: str = "Divide um texto longo em chunks, retornando um JSON que contém um dicionario com os chunks enumerados por ordem"

    def __init__(self, text_content: str, **kwargs):
        super().__init__(**kwargs)
        self._text_content = text_content


    def _run(self) -> str:
        """
        Divide o texto e retorna um JSON contendo um dicionario com chunks numerados (a partir do 1).
       : '{"1": "Primeiro chunk de texto...", "2": "Segundo chunk..."}'
        """
        try:
            # A estratégia de separadores continua a mesma para manter a coesão semântica.
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_text(self._text_content)
            
            # Filtra quaisquer chunks vazios que possam ter sido criados
            chunks = [chunk for chunk in chunks if chunk.strip()]
            
            # --- MUDANÇA PRINCIPAL AQUI ---
            # Cria um dicionário usando enumerate para obter o índice e o chunk.
            # O índice é iniciado em 1, conforme solicitado (i + 1).
            list_of_numbered_chunks = [{i + 1: chunk} for i, chunk in enumerate(chunks)]
            
            # 2. Aninha a lista no dicionário final, conforme esperado na Tarefa 4
            final_output_dict = {"json_chunks": list_of_numbered_chunks}
            
            return json.dumps(final_output_dict, ensure_ascii=False)
        except Exception as e:
            return f"Erro ao dividir o texto em um dicionário: {e}"

