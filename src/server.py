"""Custom HTTP server for the Semantic Document Indexer Web Interface.
This module has been refactored into separate modules:
- src/server/handler.py: Request handlers
- src/server/response.py: Response helpers
- src/server/__init__.py: Server initialization
"""

from __future__ import annotations

import os
import sys

from src.server.handler import DocumentHandler
from src.server.response import ErrorResponse, HTMLResponse, JSONResponse, SuccessResponse
from . import run_server


if __name__ == "__main__":
    run_server()
