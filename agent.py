import json
import os
import logging
import re
import requests
from typing import List
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class Recommendation(BaseModel):
    name: str = Field(description="Name of the assessment")
    url: str = Field(description="URL from the catalog link")
    test_type: str = Field(description="Single letter test type (e.g., P for Personality, K for Knowledge, A for Ability)")

class AgentResponse(BaseModel):
    reply: str = Field(description="The conversational reply to the user")
    recommendations: List[Recommendation] = Field(description="List of 1 to 10 recommended assessments. Empty if clarifying, refusing, or still gathering context.")
    end_of_conversation: bool = Field(description="True ONLY if the agent considers the task complete and the final shortlist is presented and accepted.")

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reply": {"type": "STRING", "description": "The conversational reply to the user"},
        "recommendations": {
            "type": "ARRAY",
            "description": "List of 1 to 10 recommended assessments.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "url": {"type": "STRING"},
                    "test_type": {"type": "STRING"}
                },
                "required": ["name", "url", "test_type"]
            }
        },
        "end_of_conversation": {"type": "BOOLEAN"}
    },
    "required": ["reply", "recommendations", "end_of_conversation"]
}

SYSTEM_PROMPT_TEMPLATE = """You are the Conversational HR Assessment Recommender.
Your job is to guide users from a vague intent to a grounded shortlist of HR assessments.

CORE RULES:
1. CLARIFY: If the query is vague, ask clarifying questions. Recommendations MUST be empty during clarification.
2. RECOMMEND: Once you have enough context, recommend 1 to 10 assessments.
3. NO HALLUCINATION: You MUST ONLY recommend assessments from the provided 'Catalog Candidates' below.
4. TEST TYPE: 'P' (Personality), 'K' (Knowledge), 'A' (Ability), 'S' (Situational).
5. END CONVERSATION: Set end_of_conversation to true ONLY when the task is complete.
6. IMPORTANT: You MUST respond ONLY with a valid JSON object. No extra text, no markdown, no explanation outside JSON.

CATALOG CANDIDATES:
{candidates_json}

Your response must be exactly this JSON structure:
{{
  "reply": "your conversational reply here",
  "recommendations": [],
  "end_of_conversation": false
}}
"""

class Agent:
    def __init__(self, retriever):
        self.retriever = retriever
        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.gemini_key = os.environ.get("GEMINI_API_KEY")

        if self.groq_key:
            self.api_provider = "groq"
            logger.info("Using Groq API.")
        elif self.gemini_key:
            self.api_provider = "gemini"
            logger.info("Using Gemini API.")
        else:
            logger.error("No API keys found! Set GROQ_API_KEY or GEMINI_API_KEY.")

    def _call_groq(self, system_instruction, messages):
        headers = {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "system", "content": system_instruction}] + messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r'\{.*\}', content, re.DOTALL)
        return match.group(0) if match else content

    def _call_gemini(self, system_instruction, messages):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self.gemini_key)
        contents = [types.Content(
            role="user" if m["role"].lower() in ["user", "human"] else "model",
            parts=[types.Part.from_text(text=m["content"])]
        ) for m in messages]
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA
        )
        response = client.models.generate_content(model='gemini-1.5-flash', contents=contents, config=config)
        return response.text

    def chat(self, messages: List[dict]) -> AgentResponse:
        # Normalize messages for API consistency
        norm_messages = []
        for m in messages:
            r = m["role"].lower()
            role = "user" if r in ["user", "human"] else "assistant"
            norm_messages.append({"role": role, "content": m["content"]})

        search_query = " ".join([m["content"] for m in norm_messages[-2:]])
        candidates = self.retriever.search(search_query, top_k=15)
        candidates_json = json.dumps(candidates, indent=2)
        system_instruction = SYSTEM_PROMPT_TEMPLATE.format(candidates_json=candidates_json)

        try:
            if self.api_provider == "groq":
                res_text = self._call_groq(system_instruction, norm_messages)
            else:
                res_text = self._call_gemini(system_instruction, norm_messages)
            return AgentResponse(**json.loads(res_text))
        except Exception as e:
            logger.error(f"Error: {e}")
            return AgentResponse(
                reply="Sorry, I'm having trouble connecting right now.",
                recommendations=[],
                end_of_conversation=False
            )