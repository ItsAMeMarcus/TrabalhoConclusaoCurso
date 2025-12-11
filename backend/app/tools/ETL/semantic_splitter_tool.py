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


    def _merge_text_chunks(self, lista_chunks: list[str], min_chars: int = 200) -> list[str]:
        chunks_finais = []
        buffer = ""  # Nossa "sala de espera" para textos curtos
    
        for chunk in lista_chunks:
            # 1. Se tem algo esperando no buffer, colamos antes do chunk atual
            if buffer:
                chunk = buffer + "\n" + chunk
                buffer = "" # Esvazia o buffer

            # 2. Verifica se o chunk (agora possivelmente maior) atingiu o tamanho mínimo
            if len(chunk) < min_chars:
                # Se ainda for pequeno, guarda no buffer para o próximo
                buffer = chunk
            else:
                # Se o tamanho está bom, aprova e adiciona na lista final
                chunks_finais.append(chunk)

        # 3. Limpeza final: Se sobrou algo no buffer (ex: o texto acabou num título)
        if buffer:
            if chunks_finais:
                # Cola no último chunk aprovado
                chunks_finais[-1] += "\n" + buffer
            else:
                # Se só tinha isso no texto todo, adiciona assim mesmo
                chunks_finais.append(buffer)

        return chunks_finais

    def _run(self) -> str:
        """
        Divide o texto e retorna um JSON contendo um dicionario com chunks numerados (a partir do 1).
       : '{"1": "Primeiro chunk de texto...", "2": "Segundo chunk..."}'
        """
        try:
            # A estratégia de separadores continua a mesma para manter a coesão semântica.
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False
            )
            chunks_bruto = text_splitter.split_text(self._text_content)
            
            chunks = self._merge_text_chunks(lista_chunks=chunks_bruto)

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
    
