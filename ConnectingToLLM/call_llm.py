from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
import os

load_dotenv()

model = init_chat_model("gemini-2.5-flash",model_provider="google_genai")
messages = [
    SystemMessage(content="You are an intelligent and helpful assistant, and try to give fact based answer to the questions you are asked.\
                  If you don't know the answer, then you will say you don't know the answer. Please answer the following question."),
    HumanMessage(content="What is the meaning of first principle thinking?"),
]
answer = model.invoke(messages)
print(answer.content)