from google.genai import types
from pydantic import BaseModel
from typing import List

class Recommendation(BaseModel):
    name: str

class AgentResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]

try:
    config = types.GenerateContentConfig(
        system_instruction="Test",
        response_mime_type="application/json",
        response_schema=AgentResponse,
        temperature=0.2,
    )
    print("Config works!")
except Exception as e:
    print(f"Error: {e}")
