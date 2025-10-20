# app/workers/celery_app.py

from celery import Celery

# O primeiro argumento é o nome do módulo principal do seu projeto.
# O 'broker' é o endereço do seu servidor Redis.
# O 'backend' também usa o Redis para armazenar os resultados das tarefas.
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
    # A linha abaixo diz ao Celery para procurar por tarefas no seu arquivo crew_tasks
    include=['app.workers.crew_assemble']
)

# Configurações opcionais
celery_app.conf.update(
    task_track_started=True,
)

# # app/workers/celery_app.py

# from celery import Celery

# # O primeiro argumento é o nome do módulo principal do seu projeto.
# # O 'broker' é o endereço do seu servidor Redis.
# # O 'backend' também usa o Redis para armazenar os resultados das tarefas.
# celery_app = Celery(
#     'tasks',
#     broker='redis://localhost:6379/0',
#     backend='redis://localhost:6379/0',
#     # A linha abaixo diz ao Celery para procurar por tarefas no seu arquivo crew_tasks
#     include=['app.workers.crew_tasks']
# )

# # Configurações opcionais
# celery_app.conf.update(
#     task_track_started=True,
# )