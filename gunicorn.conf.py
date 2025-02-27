# gunicorn.conf.py
workers = 1
timeout = 300  # Aumenta el tiempo de espera a 5 minutos
bind = "0.0.0.0:$PORT"