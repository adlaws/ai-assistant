"""Quick debug test for endpoints"""
from fastapi.testclient import TestClient
from src.api.endpoints import app

client = TestClient(app)
response = client.get("/documents")
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
print(f"JSON: {response.json() if response.status_code == 200 else 'N/A'}")
