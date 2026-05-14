import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys

from retriever import get_retriever
from agent import Agent, AgentResponse

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

app = FastAPI(title="Conversational SHL Assessment Recommender")

# Global instances
retriever = None
agent = None

@app.on_event("startup")
async def startup_event():
    global retriever, agent
    logger.info("Initializing vector database on startup...")
    retriever = get_retriever()
    agent = Agent(retriever=retriever)
    logger.info("Startup complete.")

class MessageItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[MessageItem]

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=AgentResponse)
async def chat_endpoint(request: ChatRequest):
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    # Convert Pydantic models to dicts
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    if not messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")
        
    try:
        response = agent.chat(messages)
        return response
    except Exception as e:
        logger.error(f"Error during chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
