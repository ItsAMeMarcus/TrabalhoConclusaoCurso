import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import LLM


# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configurações da API ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("A variável de ambiente GOOGLE_API_KEY não foi definida.")

# --- Instâncias de LLM pré-configuradas ---

# LLM rápido e econômico para tarefas de revisão e segmentação
# Usamos um nome genérico para facilitar a troca do modelo no futuro
llm_rapido = LLM(
    model="gemini/gemini-2.5-flash-lite",
    temperature=0.7,
)

# LLM robusto com grande janela de contexto para a tarefa final de correção/consolidação
llm_pro = LLM(
    model="gemini/gemini-2.5-pro",
    temperature=0.7,
)

# --- Configurações do Banco de Dados (exemplo) ---
db_config = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}