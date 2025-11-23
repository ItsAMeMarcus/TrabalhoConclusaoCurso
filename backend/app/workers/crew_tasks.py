from crewai import Task

from app.tools.semantic_splitter_tool import ChunksOutput

class PreProcessingTasks():
     # Tarefa 1: Dividir o texto
    def tarefa_leitura(self, preparador_texto):
        return Task(
            description=f"""
            Use a ferramentea que lhe foi dada para fazer a extração do texto do PDF e passe
            para o próximo agente
            """,
            expected_output='O texto a ser revisado, pronto para ser enviado para o revisor.',
            agent=preparador_texto
        )

    # Tarefa 2: Revisar cada pedaço
    # O 'context' aqui é crucial. Ele pega o output da tarefa anterior.
    def tarefa_revisar(self, revisor_conteudo, tarefa_leitura):
        return Task(
            description= f"""Realize uma revisão detalhada. Identifique todos os erros de gramática, ortografia, pontuação, clareza e formatação.
                ATENÇÃO ESPECIAL: LIMPEZA DE TEXTO DE GRÁFICOS.
                O texto pode conter sequências de palavras que foram extraídas de gráficos, diagramas ou fluxogramas (vetores no PDF). Essas sequências não formam frases coerentes e devem ser removidas.
                Identifique esses trechos e, no seu relatório, instrua o corretor a removê-los completamente.

                EXEMPLOS DE TEXTO PROVENIENTE DE GRÁFICO A SER REMOVIDO:
                - '57% 43% Gênero. Masculino. Feminino.'
                - '20h 30h 40h 44h CR x Jornada de trabalho. '

                IMPORTANTE: Legendas de figuras (ex: 'Figura 7. Sistemas e dos dados necessários...') geralmente são frases válidas e devem ser mantidas, mas o conjunto de rótulos desconexos que às vezes as acompanha deve ser marcado para remoção.
                Em seguida, compile um relatório consolidado com todas as sugestões. IMPORTANTE: Não corrija o texto diretamente. Seu resultado final deve ser apenas a lista de feedbacks.
                """,
            expected_output=(
                "Um único documento de texto em formato JSON. O JSON deve ser uma lista de objetos, "
                "onde cada objeto representa um erro e contém duas chaves: 'trecho_original' e 'feedback_detalhado'.\n\n"
                "Exemplo de Saída:\n"
                '[\n'
                '  {\n'
                '    "trecho_original": "...o sistema se base ada...",\n'
                '    "feedback_detalhado": "Corrigir \'base ada\' para \'baseia\'. Remover a frase \'MySQL Sistema...\' pois é um artefato de gráfico."\n'
                '  },\n'
                '  {\n'
                '    "trecho_original": "...a borda o conceito...",\n'
                '    "feedback_detalhado": "Corrigir \'a borda\' para \'aborda\'."\n'
                '  }\n'
                ']'
            ),
            agent=revisor_conteudo,
            context=[tarefa_leitura]
        )

    # Tarefa 3: Receber os feedbacks e corrigir os textos
    def tarefa_corrigir(self, corretor_feedback, tarefa_revisar, tarefa_leitura):
        return Task(
            description=(
                "Você recebeu dois insumos: o texto original e o relatório de feedbacks do revisor. "
                "Sua missão é processar uma lista de trechos do texto e seus respectivos feedbacks e aplicar cada uma das sugestões de correção ao texto. "
                "Seu resultado final deve ser o texto corrigido e limpo."
                "APLIQUE TODAS AS CORREÇÕES SUGERIDAS A TODOS OS TRECHOS E RETORNE O TEXTO COMPLETO. "
                "O seu resultado final deve ser o texto, com todos os trechos já corrigidos, formatado para NLP. "
                "Tente ser o mais eficiente possível, processando o máximo de trechos que conseguir em cada passo do seu raciocínio."
            ),
            expected_output=(
                "Sua missão tem duas fases críticas:\n\n"
                "**Fase 1: Correção do Texto**\n"
                "Você recebeu dois insumos: o texto original e um relatório de feedbacks do revisor que tem os trechos a serem corrigidos. "
                "Primeiro, aplique CADA uma das sugestões de correção e remoção aos seus respectivos pedaços de texto. "
                "Seja extremamente fiel ao feedback fornecido.\n\n"
                "**Fase 2: Formatação para NLP (BERT)**\n"
                "Após aplicar todas as correções, você deve realizar uma formatação final nos textos. "
                "O objetivo é prepará-lo para ser usado em tarefas de NLP. Siga estas regras rigorosamente:\n"
                "1. **Estrutura de Parágrafos:** Garanta que haja apenas UMA ÚNICA quebra de linha (\\n) para separar os parágrafos. Remova quebras de linha duplas ou triplas.\n"
                "2. **Fluxo das Frases:** Elimine todas as quebras de linha que ocorrem no MEIO de uma frase, unindo o texto para que cada frase seja uma linha contínua dentro do seu parágrafo.\n"
                "3. **Limpeza de Espaços:** Remova todos os espaços em branco desnecessários, como espaços duplos entre palavras ou espaços no início ou fim de uma linha.\n"
                "4. **Remoção Final de Artefatos:** Faça uma varredura final para remover quaisquer artefatos restantes que não sejam texto corrido (ex: números de página)."
                "5. **Retorno do texto completo:** Entregue como resultado o texto completamente revisado, corrigido e completo. Você não deve cortar o texto, não importa o que aconteça. Entretanto, somente o texto corrigido é interessante, ou seja, o seu processo de pensamento não pode aparecer na versão final do texto. Se for interessante, você pode pensar sobre o que vai fazer e como vai fazer e só depois retornar o texto completo"
            ),
            agent=corretor_feedback,
            context=[tarefa_leitura, tarefa_revisar]
        )

    #Tarefa 4: Dividir o texto já corrigido sem perder o sentido.
    def tarefa_segmentar(self, segmentador, tarefa_corrigir):
        return Task(
            description=(
                """Você recebeu um texto já pré-corrigido. 
                Sua missão é pegar esse texto e passá-lo diretamente como o argumento 'text_content' para a ferramenta 'Ferramenta de Divisão Semântica de Texto'. 
                Use a 'Ferramenta de Divisão Semântica de Texto' para dividir este texto completo 
                em chunks que sejam semanticamente coesos e otimizados.
                }"""
            ),
            expected_output=(
                """
                Uma ÚNICA STRING em formato JSON que representa um dicionário que vai vir diretamente da ferramenta.
                As chaves do dicionário vão ser os números dos chunks em ordem cronológica (começando em '1'), 
                e os valores vão ser os textos dos chunks. 
                Exemplo: 
                "{
                    "json_chunks":
                        [
                            {1: "Pedaco1"},
                            {2: "Pedaco2"},
                            {3: "Pedaco3"}
                        ]
                }"
                """
            ),
            agent=segmentador,
            # Esta tarefa precisa do output do corretor para começar.
            output_json=ChunksOutput,
            markdown=False,
            context=[tarefa_corrigir]
        )

    #Tarefa 5: Gerar embeddings
    def tarefa_gerar_embeddings(self, gerador_embeddings, tarefa_segmentar):
        return Task(
        description=f"""
            Gere o embedding e indexe o chunk de texto que veio do agente anterior.
            Ao receber o texto da tarefa do agente de segmentação semântica, 
    
            **Instrução Crítica:** Você DEVE usar a 'EmbeddingGeneratorTool' e passar a string que veio do agente anterior integralmente para a Tool. É de extrema importância que essa string seja entregue para a ferramenta sem nenhum tipo de corte.
            Não adicione nenhum outro texto ou histórico à chamada da Tool.
            NÃO tente ler, validar ou processar o conteúdo do JSON. 
            NÃO execute a ferramenta mais de uma vez. 
            Assim que chamar a ferramenta, considere o trabalho FEITO.
            """,        
            expected_output=(
                "Uma confirmação de salvamento."
            ),
            agent=gerador_embeddings,
            context=[tarefa_segmentar]
        )
