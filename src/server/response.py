"""Response helpers for the custom HTTP server."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler


class JSONResponse:
    """Helper class for sending JSON responses."""

    @staticmethod
    def send(handler: BaseHTTPRequestHandler, data: dict, status: int = 200) -> None:
        """Send a JSON response.

        :param handler: The HTTP request handler instance
        :param data: JSON data to send
        :param status: HTTP status code
        """
        response = json.dumps(data, indent=2)
        handler.send_response(status)
        handler.send_header('Content-Type', 'application/json')
        handler.send_header('Content-Length', len(response))
        handler.send_header('Access-Control-Allow-Origin', '*')
        handler.end_headers()
        handler.wfile.write(response.encode())


class HTMLResponse:
    """Helper class for sending HTML responses."""

    @staticmethod
    def send(handler: BaseHTTPRequestHandler, html: bytes | str, status: int = 200) -> None:
        """Send an HTML response.

        :param handler: The HTTP request handler instance
        :param html: HTML content as bytes or string
        :param status: HTTP status code
        """
        if isinstance(html, str):
            response = html.encode('utf-8')
        else:
            response = html
        handler.send_response(status)
        handler.send_header('Content-Type', 'text/html; charset=utf-8')
        handler.send_header('Content-Length', len(response))
        handler.send_header('Access-Control-Allow-Origin', '*')
        handler.end_headers()
        handler.wfile.write(response)
        handler.wfile.flush()


class ErrorResponse:
    """Helper class for sending error responses."""

    @staticmethod
    def send(handler: BaseHTTPRequestHandler, error: str, status: int = 500) -> None:
        """Send an error response.

        :param handler: The HTTP request handler instance
        :param error: Error message
        :param status: HTTP status code
        """
        JSONResponse.send(handler, {'error': error}, status)


class SuccessResponse:
    """Helper class for sending success responses."""

    @staticmethod
    def send(handler: BaseHTTPRequestHandler, message: str, status: int = 200) -> None:
        """Send a success response.

        :param handler: The HTTP request handler instance
        :param message: Success message
        :param status: HTTP status code
        """
        JSONResponse.send(handler, {'status': 'success', 'message': message}, status)
