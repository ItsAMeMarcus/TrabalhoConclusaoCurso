from fastapi import FastAPI
from app.api.endpoints import processamento, busca

# Cria a instância principal da aplicação FastAPI
app = FastAPI(title="API")

# Inclui as rotas definidas no arquivo processamento.py
# Tudo que estiver em 'processamento.router' agora estará acessível
app.include_router(processamento.router, prefix="/api/v1")
app.include_router(busca.router, prefix="/api/v1")
@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API"}
