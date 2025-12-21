# Gunicorn configuration file for Render deployment
# Handles long-running weather model requests

# Number of worker processes
workers = 2

# Worker class - sync for CPU-bound tasks
worker_class = 'sync'

# Timeout for workers (in seconds)
# Extended to 180 seconds (3 minutes) to handle HRRR data fetching
timeout = 180

# Graceful timeout (how long to wait for workers to finish before force-killing)
graceful_timeout = 180

# Keep alive connections
keepalive = 5

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Bind to port (Render uses PORT environment variable)
import os
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Preload app
preload_app = True
