import os
import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("Health check passed.")

def test_chat():
    if not os.environ.get("GEMINI_API_KEY"):
        print("Skipping chat test because GEMINI_API_KEY is not set.")
        return

    payload = {
        "messages": [
            {"role": "user", "content": "I need a coding assessment for a Java developer."}
        ]
    }
    
    print("Sending chat request...")
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "reply" in data
    assert "recommendations" in data
    assert "end_of_conversation" in data
    
    print("Chat response schema is valid!")
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    test_health()
    test_chat()
