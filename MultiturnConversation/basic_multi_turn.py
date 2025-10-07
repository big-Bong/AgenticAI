from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

load_dotenv()

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

#Define a new graph
workflow = StateGraph(state_schema=MessagesState)

#Model call
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

#Define node in graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

#Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
#Add configuration
config = {"configurable": {"thread_id": "abc123"}}

query = input("User: ")
while query != "bye":
     input_messages = [HumanMessage(query)]
     output = app.invoke({"messages":input_messages}, config)
     output["messages"][-1].pretty_print()
     query = input("User: ")