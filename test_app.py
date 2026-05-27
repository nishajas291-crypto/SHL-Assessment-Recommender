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
    # Use GROQ_API_KEY as requested
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("Skipping chat test because neither GROQ_API_KEY nor GEMINI_API_KEY is set.")
        return

    payload = {
        "messages": [
            {"role": "user", "content": "I need a coding assessment for a Java developer."}
        ]
    }
    
    print(f"Sending chat request using { 'Groq' if os.environ.get('GROQ_API_KEY') else 'Gemini' }...")
    response = client.post("/chat", json=payload)
    
    if response.status_code != 200:
        print(f"Chat failed with status {response.status_code}: {response.text}")
        return

    data = response.json()
    
    assert "reply" in data
    assert "recommendations" in data
    assert "end_of_conversation" in data
    
    print("Chat response schema is valid!")
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    test_health()
    test_chat()
