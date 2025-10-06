from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

model = init_chat_model(model="gpt-4o-mini",model_provider="openai")
system_template = "Write a program in {language} according to the instructions provided."
prompt_template = ChatPromptTemplate.from_messages(
    [("system",system_template), ("user","{text}")]
)
prompt = prompt_template.invoke({"language":"C++", "text":"Program for adding two numbers"})
response = model.invoke(prompt)
print(response.content)