from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()
model = os.environ["MODEL"]

app = FastAPI()

system_prompt = """
You are a helpful assistant that can answer questions politely.
"""

chat_history = []
llm = ChatOpenAI(model=model, temperature=0.7)


@app.get("/chat")
def chat(user_input: str):
    chat_history.append(HumanMessage(user_input))
    response = llm.invoke(chat_history)
    chat_history.append(AIMessage(response.content))
    return response.content
