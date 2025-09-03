
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from vermeg_rag_chatbot_gemini import VermegGeminiChatbot

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
    answer = chatbot.generate_response(request.question)
    return {"answer": answer}

@app.get("/")
def root():
    return FileResponse("static/index.html")
