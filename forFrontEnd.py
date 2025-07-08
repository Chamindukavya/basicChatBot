from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal


load_dotenv()
model = os.environ["MODEL"]

app = FastAPI()

class Message(BaseModel):
    type: Literal["human", "ai"]
    content: str

class ChatRequest(BaseModel):
    chat: List[Message]

system_prompt = """
You are a helpful assistant that can answer questions politely.
"""


llm = ChatOpenAI(model=model, temperature=0.7)


@app.post("/chat")
def chat(user_input: ChatRequest):
    chat_history = []
    chat_history.append(SystemMessage(system_prompt))
    for message in user_input.chat:
        if message.type == "human":
            chat_history.append(HumanMessage(message.content))
        elif message.type == "ai":
            chat_history.append(AIMessage(message.content))
        else:
            pass
    response = llm.invoke(chat_history)
    return response.content
