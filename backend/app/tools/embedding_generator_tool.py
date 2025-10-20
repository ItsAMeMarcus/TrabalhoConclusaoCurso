import json
import torch
from crewai.tools import BaseTool
from transformers import BertTokenizer, BertModel
from typing import List, Any, Dict
from pydantic.v1 import PrivateAttr

class EmbeddingGeneratorTool(BaseTool):
    name: str = "Gerador de Embeddings BERT"
    description: str = "Recebe um dicionário JSON de chunks, gera embeddings e retorna um dicionário JSON com os embeddings."

    _device: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, model_name='neuralmind/bert-base-portuguese-cased', **kwargs):
        super().__init__(**kwargs)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"EmbeddingGeneratorTool: Usando dispositivo: {self._device}")
        
        self._tokenizer = BertTokenizer.from_pretrained(model_name)
        self._model = BertModel.from_pretrained(model_name).to(self._device)
        self._model.eval()
        print(f"EmbeddingGeneratorTool: Modelo '{model_name}' carregado com sucesso.")

    def _run(self, json_chunks_dict: str) -> str:
        """
        Método principal que recebe uma STRING JSON contendo chunks,
        gera os embeddings e retorna uma nova STRING JSON com os embeddings.
        """
        try:
            # --- MUDANÇA PRINCIPAL (ENTRADA) ---
            # 1. Converte a string JSON de volta para um dicionário Python
            chunks_dict = json.loads(json_chunks_dict)
            
            # 2. Extrai apenas os textos (values) do dicionário para processamento
            chunks_texts = list(chunks_dict.values())
            
            if not chunks_texts:
                return "Erro: O dicionário de chunks recebido está vazio."

            # O resto da lógica de embedding continua a mesma...
            inputs = self._tokenizer(
                chunks_texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_pooled_embeddings = sum_embeddings / sum_mask
            
            embeddings_list = mean_pooled_embeddings.cpu().numpy().tolist()

            # --- MUDANÇA PRINCIPAL (SAÍDA) ---
            # 3. Cria um novo dicionário associando o ID original de cada chunk ao seu novo embedding
            #    Isso mantém a conexão entre o texto e seu vetor.
            embeddings_dict = {chunk_id: embedding for chunk_id, embedding in zip(chunks_dict.keys(), embeddings_list)}
            
            # 4. Retorna o resultado como uma string JSON para a próxima tarefa
            return json.dumps(embeddings_dict)

        except json.JSONDecodeError:
            return "Erro: A entrada não era uma string JSON válida."
        except Exception as e:
            return f"Ocorreu um erro ao gerar os embeddings: {e}"
