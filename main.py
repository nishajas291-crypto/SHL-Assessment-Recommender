import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys

from retriever import get_retriever
from agent import Agent, AgentResponse

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Conversational SHL Assessment Recommender")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        logger.error(f"Error during chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
