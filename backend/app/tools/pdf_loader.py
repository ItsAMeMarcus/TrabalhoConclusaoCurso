from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
import io
import PyPDF2
from typing import Any

class PDFLoaderInput(BaseModel):
    argument: str = Field(..., description="Lê um PDF e extrai o texto dele, retornando-o em uma string e passando para os próximos agentes")

class PDFLoaderTool(BaseTool):
    name: str = "Ferramenta de Extração de Texto"
    description: str = "Lê um PDF e extrai o texto dele, retornando-o em uma string e passando para os próximos agentes"
    _pdf: Any = PrivateAttr()
    def __init__(self, pdf_bytes:bytes, **kwargs):
        super().__init__(**kwargs)
        self._pdf = pdf_bytes

    def _run(self) -> str | None:
        try:
            # Cria um "arquivo virtual" em memória a partir dos bytes
            pdf_file_in_memory = io.BytesIO(self._pdf)
        
            reader = PyPDF2.PdfReader(pdf_file_in_memory)
            texto_completo = ""
            for page in reader.pages:
                texto_completo += page.extract_text() + "\n"
            return texto_completo
        except Exception as e:
            print(f"Erro ao ler o PDF a partir dos bytes: {e}")
            raise e
