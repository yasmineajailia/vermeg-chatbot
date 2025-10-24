
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from vermeg_rag_chatbot_gemini import VermegGeminiChatbot
import json

app = FastAPI()
chatbot = VermegGeminiChatbot()
chatbot.load_documents("digital solutions")
chatbot.load_documents("vermeg core solutions")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/src", StaticFiles(directory="src"), name="src")

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    """Non-streaming endpoint for backwards compatibility."""
    response = chatbot.generate_response(request.question)
    return response

@app.post("/ask/stream")
async def ask_question_stream(request: QueryRequest):
    """Streaming endpoint that sends tokens as they're generated."""
    async def generate():
        for chunk in chatbot.generate_response_stream(request.question):
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/")
def root():
    return FileResponse("static/index.html")
