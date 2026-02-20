#!/usr/bin/env python3
"""
run.py — Start the Flask API server.

For production, use gunicorn:
    gunicorn "app.app:app" -w 1 -b 0.0.0.0:5000 --timeout 120
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from app.app import app
from app.config import Config

if __name__ == "__main__":
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True,
    )
