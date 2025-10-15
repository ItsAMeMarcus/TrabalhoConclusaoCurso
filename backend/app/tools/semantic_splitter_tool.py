# app/tools/semantic_splitter_tool.py

import json
from crewai.tools import BaseTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

class SemanticTextSplitterTool(BaseTool):
    name: str = "Ferramenta de Divisão Semântica de Texto"
    description: str = "Divide um texto longo em chunks, retornando um dicionário com os chunks numerados."

    def _run(self, text_content: str) -> str:
        """
        Divide o texto e retorna um dicionário JSON com chunks numerados (a partir do 1).
        Exemplo de retorno: '{"1": "Primeiro chunk de texto...", "2": "Segundo chunk..."}'
        """
        try:
            # A estratégia de separadores continua a mesma para manter a coesão semântica.
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            chunks = text_splitter.split_text(text_content)
            
            # Filtra quaisquer chunks vazios que possam ter sido criados
            chunks = [chunk for chunk in chunks if chunk.strip()]
            
            # --- MUDANÇA PRINCIPAL AQUI ---
            # Cria um dicionário usando enumerate para obter o índice e o chunk.
            # O índice é iniciado em 1, conforme solicitado (i + 1).
            chunk_dict = {str(i + 1): chunk for i, chunk in enumerate(chunks)}
            
            # Retorna o dicionário como uma string JSON.
            # Usar ensure_ascii=False é uma boa prática para lidar com caracteres em português.

            print(json.dumps(chunk_dict, ensure_ascii=False))

            return json.dumps(chunk_dict, ensure_ascii=False)

        except Exception as e:
            return f"Erro ao dividir o texto em um dicionário: {e}"

