"""
Compatibility shim for Render/Gunicorn.

Some start commands use `gunicorn app:app`. This file ensures that works by
re-exporting the Flask app instance defined in `server.py`.
"""

from server import app  # noqa: F401

