import os
from dotenv import load_dotenv
from crewai import LLM
from langfuse import Langfuse 
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configurações da API ---
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
#LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
#LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

#if not GOOGLE_API_KEY:
 #   raise ValueError("A variável de ambiente GOOGLE_API_KEY não foi definida.")

#langfuse = Langfuse(
#    secret_key=LANGFUSE_SECRET_KEY,
 #   public_key=LANGFUSE_PUBLIC_KEY,
  #  host=LANGFUSE_HOST
#)

# --- Instâncias de LLM pré-configuradas ---
# LLM rápido e econômico para tarefas de revisão e segmentação
# Usamos um nome genérico para facilitar a troca do modelo no futuro
llm_rapido = LLM(
    model="gemini/gemini-2.0-flash-lite",
    temperature=0.2,
    top_p=0.4
)

# LLM robusto com grande janela de contexto para a tarefa final de correção/consolidação
llm_pro = LLM(
    model="gemini/gemini-2.5-pro",
    temperature=0.2,
    top_p=0.4
)

llm_flash = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.3,
    top_p=0.4
)

llm_flash_lite = LLM(
    model="gemini/gemini-2.5-flash-lite",
    temperature=0.3,
    top_p=0.4
)

# --- Configurações do Banco de Dados (exemplo) ---
db_config = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

CrewAIInstrumentor().instrument(skip_dep_check=True)
LiteLLMInstrumentor().instrument()
