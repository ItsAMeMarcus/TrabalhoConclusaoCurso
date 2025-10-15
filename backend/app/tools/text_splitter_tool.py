from langchain.text_splitter import RecursiveCharacterTextSplitter
from crewai.tools import BaseTool

class TextSplitterTool(BaseTool):
    name: str = "Text Splitter Tool"
    description: str = "Divide um texto longo em pedaços menores (chunks) para não exceder a janela de contexto da LLM."

    def _run(self, text_content: str) -> str:
        # Define o tamanho do chunk e a sobreposição (overlap)
        # Ajuste esses valores conforme o modelo que você usa (GPT-4, Llama, etc.)
        # Ex: Para GPT-4 (8k context), chunks de 4000 são seguros.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=25, # A sobreposição ajuda a manter o contexto entre os chunks
            length_function=len
        )
        chunks = text_splitter.split_text(text_content)
        return chunks