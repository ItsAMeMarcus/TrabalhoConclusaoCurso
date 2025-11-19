import json  # 1. Importe a biblioteca JSON
from crewai.tasks.task_output import TaskOutput

def extrair_payload_do_log(output: TaskOutput):
    """
    Função de callback para extrair o payload de um log JSON
    armazenado em output.raw.
    """
    raw_output = output.raw  # Esta variável contém a string JSON
    
    try:
        # 2. Converta a string JSON em um dicionário Python
        log_data = json.loads(raw_output)
        
        # 3. Acesse o payload que você deseja
        # Usar .get() é mais seguro, pois retorna None se a chave não existir,
        # em vez de quebrar o código.
        
        # Opção A: Pegar o payload do evento final (geralmente o resultado)
        end_event_payload = log_data.get("endEvent", {}).get("data", {}).get("payload")
        
        if end_event_payload:
            print("\n[MIDDLEWARE] Payload do 'endEvent' extraído com sucesso.")
            
            # 'end_event_payload' agora é um dicionário Python.
            # Você pode acessar qualquer coisa dentro dele:
            print(f"Ferramenta usada: {end_event_payload.get('tool_name')}")
            print(f"Duração: {log_data.get('duration')}ms")
            
            # 4. Retorne o payload (agora como um dicionário)
            return end_event_payload

        # Opção B: Pegar o payload do evento inicial (tem a config do agente)
        # start_event_payload = log_data.get("startEvent", {}).get("data", {}).get("payload")
        # if start_event_payload:
        #     print(f"Agente: {start_event_payload.get('agent', {}).get('role')}")
        #     return start_event_payload

        else:
            print("\n[MIDDLEWARE] Payload não encontrado dentro de 'endEvent.data'.")
            return None
            
    except json.JSONDecodeError:
        # Isso acontece se o 'output.raw' não for um JSON válido
        print(f"\n[MIDDLEWARE] Erro: O 'output.raw' não é um JSON válido.")
        return None
    except Exception as e:
        print(f"\n[MIDDLEWARE] Erro inesperado ao processar o JSON: {e}")
        return None

