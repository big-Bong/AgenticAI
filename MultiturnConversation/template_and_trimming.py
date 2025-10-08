#Multi turn conversations with prompt template and maintaining limited conversation history

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState ,StateGraph

load_dotenv()

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a good therapist and like talking to people and making them comfortable. You try to understand \
            what a person is going through by talking with them."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

query = input("User: ")
while query != "bye":
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages":input_messages}, config)
    output["messages"][-1].pretty_print()
    query = input("User: ")


